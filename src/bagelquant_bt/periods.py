"""Read-only aggregation over saved BT primitives and caller-selected periods."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date

import polars as pl


def slice_primitives(frames, start: date, end: date):
    """Apply observation and label-maturity bounds, never include future labels."""
    result = {}
    for name, frame in frames.items():
        if name.startswith(("table_", "period_")):
            continue
        column = next(
            (
                item
                for item in ("evaluation_date", "time", "fit_date")
                if item in frame.columns
            ),
            None,
        )
        if column:
            frame = frame.filter(pl.col(column).is_between(start, end))
        for maturity in (
            "target_end_date",
            "target_date",
            "interval_end",
            "available_date",
        ):
            if maturity in frame.columns:
                frame = frame.filter(pl.col(maturity) <= end)
        result[name] = frame
    return result


def aggregate_saved_window(kind, section, frames, settings, *, start, end, items=()):
    """Recompute metrics/tables for a period without producing or persisting values."""
    from bagelquant_bt.sample import evaluation_sample
    from bagelquant_bt.statistics import return_statistics
    from bagelquant_bt.window import compute_window_tables

    requested_start = start
    metadata = (
        evaluation_sample(frames, start=start, end=end) if kind != "portfolio" else None
    )
    if metadata and metadata["effective_start"]:
        start = date.fromisoformat(metadata["effective_start"])
    sample = slice_primitives(frames, start, end)
    if (
        metadata
        and metadata["effective_start"] is None
        and "signal_count"
        in frames.get("daily_signal_coverage", pl.DataFrame()).columns
        and section not in {"input_evidence", "model_diagnostics"}
    ):
        sample = {name: frame.head(0) for name, frame in sample.items()}
    annualization = settings["annualization"]
    if section == "comparison":
        from bagelquant_bt import hac_mean_test

        horizon = int(settings.get("comparison_protocol", {}).get("main_horizon", 5))
        increments = {
            name: {
                column: asdict(
                    hac_mean_test(
                        frame[column].to_list(),
                        window_width=horizon if name == "comparison_ic" else 1,
                    )
                )
                for column in frame.columns
                if column.endswith("_increment") or column == "conditional_ic"
            }
            for name, frame in sample.items()
            if "time" in frame.columns
        }
        return {
            "status": "ready",
            "baseline": settings.get("comparison_baseline"),
            "increments": increments,
            "common_sample_counts": {
                name: frame.height for name, frame in sample.items()
            },
        }, sample
    if section == "implementation_stress":
        rows = []
        for name, performance in sample.items():
            if not name.startswith("stress_") or not name.endswith(
                "_account_performance"
            ):
                continue
            scenario = name.removeprefix("stress_").removesuffix("_account_performance")
            statistics = return_statistics(
                performance.rename({"unit_return": "net_return"}),
                annualization=annualization,
            )
            capacity = sample.get("stress_" + scenario + "_capacity", pl.DataFrame())
            declared = frames.get("stress_summary", pl.DataFrame())
            parameters = (
                declared.filter(pl.col("scenario") == scenario)
                .select("initial_capital", "session_lag", "commission", "buy_slippage")
                .row(0, named=True)
                if "initial_capital" in declared.columns
                else {}
            )
            rows.append(
                {
                    "scenario": scenario,
                    **parameters,
                    **{
                        key: value
                        for key, value in statistics.items()
                        if not isinstance(value, (list, dict))
                    },
                    "capacity_p50": capacity["participation"].quantile(0.5)
                    if "participation" in capacity.columns
                    else None,
                    "capacity_p90": capacity["participation"].quantile(0.9)
                    if "participation" in capacity.columns
                    else None,
                    "capacity_unavailable": capacity["participation"].null_count()
                    if "participation" in capacity.columns
                    else 0,
                }
            )
        sample["stress_summary"] = pl.DataFrame(rows)
        return {
            "status": "ready",
            "scenarios": rows,
            "capacity_unit": "CNY",
            "capacity_window": "strict_prior_20_sessions",
        }, sample
    if kind == "portfolio":
        if section == "execution":
            from .execution_evaluation import summarize_transaction_pnl

            account_frames = {
                name.removeprefix("account_"): frame
                for name, frame in frames.items()
                if name.startswith("account_")
            }
            sample["account_transaction_pnl"] = summarize_transaction_pnl(
                account_frames, through=end
            )
            sessions = account_frames["account_value"].filter(pl.col("time") <= end)
            if not sessions.is_empty():
                cutoff = sessions["time"].max()
                for ledger in ("", "ideal_"):
                    history = account_frames[ledger + "open_lot_history"]
                    sample["account_" + ledger + "open_lots"] = history.filter(
                        pl.col("time") == cutoff
                    ).drop("time")
            for name in ("account_fifo_matches",):
                if "exit_date" in sample[name].columns:
                    sample[name] = sample[name].filter(
                        pl.col("exit_date").is_between(start, end)
                    )
        metrics = {}
        if section in {"summary", "performance"}:
            performance = sample.get("account_performance", pl.DataFrame())
            metrics["account"] = return_statistics(
                performance.rename({"unit_return": "net_return"}),
                annualization=annualization,
            )
        if section == "benchmark":
            from bagelquant_bt.benchmarks import compare_portfolio_to_benchmarks

            returns = (
                sample["benchmark_comparison"]
                .select("time", "portfolio_return")
                .unique()
                .rename({"portfolio_return": "return"})
            )
            path, statistics = compare_portfolio_to_benchmarks(
                returns,
                sample["benchmark_returns"].select("time", "benchmark", "return"),
                annualization=annualization,
            )
            sample.update(benchmark_comparison=path, benchmark_statistics=statistics)
            metrics["benchmark_contract"] = "saved-value-benchmark.v1"
        # Rebase the saved account's wealth; preserve equity and actual holdings.
        account = sample.get("account_account_value")
        if (
            account is not None
            and not account.is_empty()
            and "nav" in account.columns
            and account["nav"][0] > 0
        ):
            performance = sample.get("account_performance", pl.DataFrame())
            if "unit_return" in performance.columns:
                wealth = performance.sort("time").select(
                    "time",
                    (1 + pl.col("unit_return").fill_null(0)).cum_prod().alias("nav"),
                )
                sample["account_account_value"] = account.drop("nav").join(
                    wealth, on="time", how="left"
                )
        return metrics, sample
    if section == "model_diagnostics":
        audit = sample.get("training_audits", pl.DataFrame())
        status = (
            "ready"
            if "training_audits" in sample
            else "requires_rebuild"
            if settings.get("model_audit_required")
            else "not_applicable"
        )
        return {"status": status, "fit_count": audit.height}, sample
    sample["horizon_signal_persistence"] = sample.get(
        "daily_signal_autocorrelation", pl.DataFrame()
    )
    metrics, tables = compute_window_tables(
        section,
        items,
        returns=pl.DataFrame(),
        turnover=pl.DataFrame(),
        costs=pl.DataFrame(),
        series=sample,
        annualization=annualization,
        ic_annualization=annualization,
    )
    for label, name in (("book", "daily_book_returns"), ("tail", "daily_tail_returns")):
        if name in sample:
            metrics[label] = return_statistics(
                sample[name], annualization=annualization
            )
    metrics["sample"] = metadata
    if section in {"summary", "ic_horizon_profile"}:
        tables.update(
            _research_summary_tables(sample, annualization=annualization, end=end)
        )
        if "daily_signal_coverage" in frames:
            tables["signal_coverage"] = frames["daily_signal_coverage"].filter(
                pl.col("time").is_between(requested_start, end)
            )
    return metrics, tables


def _research_summary_tables(frames, *, annualization, end):
    """Expose annual/recent and coverage evidence from saved economic samples."""
    from .horizon import summarize_window_ic
    from .statistics import return_statistics

    annual, recent, coverage = [], [], []
    for label in ("book", "tail"):
        frame = frames.get(f"daily_{label}_returns", pl.DataFrame())
        if "time" not in frame.columns or frame.is_empty():
            continue
        for (year,), part in frame.group_by(pl.col("time").dt.year().alias("year")):
            annual.append(
                {
                    "portfolio": label,
                    "year": year,
                    "start": part["time"].min(),
                    "end": part["time"].max(),
                    **return_statistics(part, annualization=annualization),
                }
            )
        part = frame.filter(pl.col("time") >= date(end.year - 4, 1, 1))
        recent.append(
            {
                "portfolio": label,
                "start": part["time"].min(),
                "end": part["time"].max(),
                **return_statistics(part, annualization=annualization),
            }
        )
    ic = frames.get("horizon_ic", pl.DataFrame())
    annual_ic = []
    if not ic.is_empty():
        for (year,), part in ic.group_by(
            pl.col("evaluation_date").dt.year().alias("year")
        ):
            annual_ic.append(
                summarize_window_ic(
                    part, annualization_sessions=annualization
                ).with_columns(pl.lit(year).alias("year"))
            )
    signals = frames.get("daily_signal_coverage", pl.DataFrame())
    if {"signal_count", "universe_count"} <= set(
        signals.columns
    ) and not signals.is_empty():
        ratios = (signals["signal_count"] / signals["universe_count"]).filter(
            signals["universe_count"] > 0
        )
        coverage.append(
            {
                "kind": "signal",
                "window_id": None,
                "sample_size": ratios.len(),
                "median": ratios.median(),
                "p10": ratios.quantile(0.1, interpolation="linear"),
            }
        )
    forward = frames.get("horizon_coverage", pl.DataFrame())
    if "coverage_ratio" in forward.columns:
        for (window,), part in forward.group_by("window_id"):
            ratios = part["coverage_ratio"].drop_nulls()
            coverage.append(
                {
                    "kind": "forward_return",
                    "window_id": window,
                    "sample_size": ratios.len(),
                    "median": ratios.median(),
                    "p10": ratios.quantile(0.1, interpolation="linear"),
                }
            )
    tables = {}
    for name, rows, ordering in (
        ("annual_performance", annual, ["portfolio", "year"]),
        ("recent_performance", recent, ["portfolio"]),
        ("coverage_summary", coverage, ["kind", "window_id"]),
    ):
        if rows:
            tables[name] = pl.DataFrame(rows, infer_schema_length=None).sort(ordering)
    if annual_ic:
        tables["annual_ic"] = pl.concat(annual_ic).sort(
            ["year", "method", "window_kind", "end_session"]
        )
        recent_start = date(end.year - 4, 1, 1)
        tables["recent_ic"] = summarize_window_ic(
            ic.filter(pl.col("evaluation_date") >= recent_start),
            annualization_sessions=annualization,
        ).with_columns(
            pl.lit(max(recent_start, ic["evaluation_date"].min())).alias("start"),
            pl.lit(end).alias("end"),
        )
    return tables
