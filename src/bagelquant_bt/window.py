"""Fast date-window aggregation over cached factor and portfolio primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import polars as pl

from ._quantiles import sort_quantile_frame
from .benchmarks import benchmark_performance, top_n_excess_returns
from .horizon import (
    SIGNAL_PERSISTENCE_HORIZONS,
    build_statistical_inference,
    summarize_signal_persistence,
    summarize_window_ic,
)
from .inputs import TIME
from .performance import _annualized_return, rolling_performance
from .statistics import one_sample_t_test, quantile_rank_information_coefficients


def compute_window_tables(
    section: str,
    items: tuple[str, ...],
    *,
    returns: pl.DataFrame,
    turnover: pl.DataFrame,
    costs: pl.DataFrame,
    series: Mapping[str, pl.DataFrame],
    annualization: int,
    ic_annualization: int,
    periods: pl.DataFrame | None = None,
    benchmark_returns: pl.DataFrame | None = None,
) -> tuple[dict[str, bool | float | int | None], dict[str, pl.DataFrame]]:
    """Build one result section from already-windowed primitive series."""

    selected = set(items)
    horizon_protocol = "horizon_ic" in series
    daily_summary_items = {
        "signal_coverage",
        "daily_turnover",
        "signal_autocorrelation",
        "implied_half_life",
        "sharpe_lead_lag",
        "cumulative_return",
    }
    if section == "summary" and selected & daily_summary_items:
        return {}, _daily_summary_tables(series, selected)
    if section == "summary" and horizon_protocol:
        return {}, _horizon_summary_tables(
            series,
            annualization_sessions=ic_annualization,
        )
    if section == "ic_horizon_profile":
        ic = series.get("horizon_ic", pl.DataFrame())
        tables = {
            "ic_horizon": ic,
            "ic_horizon_summary": summarize_window_ic(
                ic,
                annualization_sessions=ic_annualization,
            ),
        }
        if "rolling_ic" in selected:
            tables["rolling_ic"] = series.get("daily_rolling_ic", pl.DataFrame())
        return {}, tables
    if section == "alpha_return":
        return {}, {
            "alpha_return_lag_returns": series.get(
                "daily_alpha_return_lag_returns", pl.DataFrame()
            ),
            "book_daily_returns": series.get("daily_book_returns", pl.DataFrame()),
            "tail_daily_returns": series.get("daily_tail_returns", pl.DataFrame()),
        }
    if section == "quantile_test":
        return {}, _daily_quantile_test_tables(
            series.get("daily_quantile_returns", pl.DataFrame()),
            annualization=annualization,
        )
    if section == "book_tail_quantiles":
        tables: dict[str, pl.DataFrame] = {}
        for item, primitive in (
            ("book_returns", "horizon_book_returns"),
            ("tail_returns", "horizon_tail_returns"),
            ("quantile_curve", "horizon_quantile_forward_returns"),
            ("quantile_structure", "horizon_quantile_structure"),
        ):
            if item in selected:
                tables[item] = series.get(primitive, pl.DataFrame())
        return {}, tables
    if section == "signal_persistence":
        persistence = series.get("horizon_signal_persistence", pl.DataFrame())
        return {}, {
            "signal_persistence": persistence,
            "signal_persistence_summary": summarize_signal_persistence(
                persistence,
                SIGNAL_PERSISTENCE_HORIZONS,
            ),
        }
    if section == "stability":
        tables = {}
        if "period_comparison" in selected:
            tables["stability"] = series.get("horizon_stability", pl.DataFrame())
        if "rolling" in selected:
            tables["rolling_stability"] = series.get(
                "horizon_rolling_stability", pl.DataFrame()
            )
        return {}, tables
    if section == "statistical_tests" and horizon_protocol:
        return {}, {
            "statistical_inference": build_statistical_inference(
                ic=series.get("horizon_ic", pl.DataFrame()),
                book_returns=series.get("horizon_book_returns", pl.DataFrame()),
                tail_returns=series.get("horizon_tail_returns", pl.DataFrame()),
                quantile_structure=series.get(
                    "horizon_quantile_structure", pl.DataFrame()
                ),
                factor_returns=series.get("horizon_factor_returns", pl.DataFrame()),
            )
        }
    if section == "summary":
        metrics, tables = _summary(
            returns,
            turnover,
            series,
            annualization,
            ic_annualization,
            benchmark_returns,
            periods,
        )
        tables["bankruptcies"] = _bankruptcies(
            returns=returns,
            lag_returns=series.get("lag_returns", pl.DataFrame()).filter(
                (pl.col("portfolio") == "spread") & (pl.col("lag") == 0)
            )
            if not series.get("lag_returns", pl.DataFrame()).is_empty()
            else pl.DataFrame(),
        )
        return metrics, tables
    if section == "ic":
        return {}, _ic_tables(series, selected, ic_annualization, periods)
    if section == "spread":
        tables = _spread_tables(series, selected, annualization)
        tables["bankruptcies"] = _bankruptcies(
            lag_returns=series.get("lag_returns", pl.DataFrame()).filter(
                pl.col("portfolio") == "spread"
            )
            if not series.get("lag_returns", pl.DataFrame()).is_empty()
            else pl.DataFrame()
        )
        return {}, tables
    if section == "top_n":
        tables = _top_n_tables(
            returns,
            series,
            selected,
            annualization,
            benchmark_returns,
        )
        tables["bankruptcies"] = _bankruptcies(
            returns=returns,
            lag_returns=series.get("lag_returns", pl.DataFrame()).filter(
                pl.col("portfolio") == "top_n"
            )
            if not series.get("lag_returns", pl.DataFrame()).is_empty()
            else pl.DataFrame(),
        )
        return {"benchmark_available": benchmark_returns is not None}, tables
    if section == "quantiles":
        tables = _quantile_tables(series, selected, ic_annualization)
        tables["bankruptcies"] = _bankruptcies(
            quantile_returns=series.get("quantile_returns", pl.DataFrame())
        )
        return {}, tables
    if section == "statistical_tests":
        lag_returns = series.get("lag_returns", pl.DataFrame())
        return {}, {
            "statistical_tests": _statistical_tests(series, periods),
            "bankruptcies": _bankruptcies(
                lag_returns=lag_returns.filter(
                    (pl.col("portfolio") == "spread") & (pl.col("lag") == 0)
                )
                if not lag_returns.is_empty()
                else pl.DataFrame()
            ),
        }
    raise ValueError(f"unknown result section: {section}")


def _daily_quantile_test_tables(
    frame: pl.DataFrame,
    *,
    annualization: int,
) -> dict[str, pl.DataFrame]:
    """Build continuous daily-rebalanced gross quantile portfolio paths."""

    required = {TIME, "quantile", "gross_return"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return {}
    returns = frame.select(
        TIME,
        "quantile",
        pl.col("gross_return").cast(pl.Float64).alias("return"),
        *(["constituent_count"] if "constituent_count" in frame.columns else []),
        *(["unavailable_reason"] if "unavailable_reason" in frame.columns else []),
    ).sort(["quantile", TIME])
    if returns.is_empty():
        return {}
    quantile_count = returns.get_column("quantile").n_unique()
    common_dates = (
        returns.group_by(TIME)
        .agg(
            pl.col("quantile").n_unique().alias("_quantile_count"),
            (pl.col("return").is_not_null() & pl.col("return").is_finite())
            .all()
            .alias("_all_finite"),
        )
        .filter((pl.col("_quantile_count") == quantile_count) & pl.col("_all_finite"))
        .select(TIME)
    )
    path = (
        sort_quantile_frame(returns, after=(TIME,))
        .with_columns(
            pl.when(pl.col("return").is_not_null() & pl.col("return").is_finite())
            .then(1.0 + pl.col("return"))
            .otherwise(None)
            .alias("_growth")
        )
        .with_columns(
            (pl.col("_growth").cum_prod().over("quantile") - 1.0).alias(
                "cumulative_return"
            )
        )
        .drop("_growth")
    )
    performance_returns = returns.join(common_dates, on=TIME, how="inner")
    if performance_returns.is_empty():
        return {
            "quantile_test_returns": path,
            "quantile_test_performance": pl.DataFrame(
                schema={
                    "quantile": pl.String,
                    "annualized_return": pl.Float64,
                    "observation_count": pl.Int64,
                }
            ),
        }
    rows = []
    for sample in performance_returns.partition_by("quantile", maintain_order=True):
        values = sample.get_column("return")
        rows.append(
            {
                "quantile": sample.item(0, "quantile"),
                "annualized_return": _single_return_metrics(values, annualization)[
                    "annualized_return"
                ],
                "observation_count": values.len(),
            }
        )
    performance = sort_quantile_frame(pl.DataFrame(rows))
    return {
        "quantile_test_returns": path,
        "quantile_test_performance": performance,
    }


def _daily_summary_tables(
    series: Mapping[str, pl.DataFrame],
    selected: set[str],
) -> dict[str, pl.DataFrame]:
    tables: dict[str, pl.DataFrame] = {}
    if "signal_coverage" in selected:
        tables["signal_coverage"] = series.get("daily_signal_coverage", pl.DataFrame())
    if "daily_turnover" in selected:
        tables["daily_turnover"] = series.get("daily_book_turnover", pl.DataFrame())
    if {"signal_autocorrelation", "implied_half_life"} & selected:
        tables["signal_autocorrelation"] = series.get(
            "daily_signal_autocorrelation", pl.DataFrame()
        )
    if "sharpe_lead_lag" in selected:
        tables["lead_lag_returns"] = series.get(
            "daily_book_lead_lag_returns", pl.DataFrame()
        )
    if "cumulative_return" in selected:
        tables["book_daily_returns"] = series.get("daily_book_returns", pl.DataFrame())
        tables["tail_daily_returns"] = series.get("daily_tail_returns", pl.DataFrame())
    return tables


def _horizon_summary_tables(
    series: Mapping[str, pl.DataFrame],
    *,
    annualization_sessions: int,
) -> dict[str, pl.DataFrame]:
    ic = series.get("horizon_ic", pl.DataFrame())
    ic_summary = summarize_window_ic(
        ic,
        annualization_sessions=annualization_sessions,
    )
    keys = ["window_kind", "window_id", "start_session", "end_session"]
    horizon = ic.select(*keys).unique() if not ic.is_empty() else pl.DataFrame()
    for method in ("pearson", "spearman"):
        selected = ic_summary.filter(pl.col("method") == method).select(
            *keys,
            pl.col("mean").alias(f"{method}_ic"),
            pl.col("icir").alias(f"{method}_icir"),
            pl.col("positive_ratio").alias(f"{method}_positive_ratio"),
        )
        horizon = (
            selected
            if horizon.is_empty()
            else horizon.join(selected, on=keys, how="left")
        )
    for primitive, value_column, output in (
        ("horizon_book_returns", "book_return", "book_return"),
        ("horizon_tail_returns", "tail_return", "tail_return"),
        ("horizon_coverage", "coverage_ratio", "coverage_ratio"),
    ):
        frame = series.get(primitive, pl.DataFrame())
        if frame.is_empty():
            continue
        summary = frame.group_by(*keys).agg(pl.col(value_column).mean().alias(output))
        horizon = (
            summary
            if horizon.is_empty()
            else horizon.join(summary, on=keys, how="left")
        )
    return {
        "horizon_summary": horizon.sort(["window_kind", "end_session"])
        if not horizon.is_empty()
        else horizon,
        "coverage": series.get("horizon_coverage", pl.DataFrame()),
    }


def _summary(
    returns: pl.DataFrame,
    turnover: pl.DataFrame,
    series: Mapping[str, pl.DataFrame],
    annualization: int,
    ic_annualization: int,
    benchmark_returns: pl.DataFrame | None,
    periods: pl.DataFrame | None,
) -> tuple[dict[str, bool | float | int | None], dict[str, pl.DataFrame]]:
    top_n = _paired_return_metrics(returns, annualization)
    spread = _lag_period_returns(series, portfolio="spread", lag=0)
    spread_metrics = _paired_return_metrics(spread, annualization)
    ic = _ic_frame(series, periods)
    row: dict[str, bool | float | int | None] = {}
    for method, column in (("pearson", "pearson_ic"), ("spearman", "spearman_ic")):
        values = ic.get_column(column).drop_nulls() if column in ic.columns else []
        summary = _ic_summary(values, ic_annualization)
        row[f"{method}_ic"] = summary["mean"]
        row[f"{method}_icir"] = summary["icir"]
        row[f"{method}_ic_p_value"] = one_sample_t_test(values).p_value
    row.update(
        {
            "spread_net_annualized_return": spread_metrics["net_annualized_return"],
            "spread_net_sharpe": spread_metrics["net_sharpe"],
            "spread_net_annualized_volatility": spread_metrics[
                "net_annualized_volatility"
            ],
            "spread_net_max_drawdown": spread_metrics["net_max_drawdown"],
            "spread_net_calmar": spread_metrics["net_calmar"],
            "top_n_net_annualized_return": top_n["net_annualized_return"],
            "top_n_net_annualized_excess_return": (
                _annualized_relative_wealth_excess(
                    returns,
                    benchmark_returns,
                    annualization,
                )
            ),
            "top_n_net_sharpe": top_n["net_sharpe"],
            "top_n_net_annualized_volatility": top_n["net_annualized_volatility"],
            "top_n_net_max_drawdown": top_n["net_max_drawdown"],
            "top_n_net_calmar": top_n["net_calmar"],
            "top_n_annualized_turnover": _annualized_turnover(turnover, annualization),
            "top_n_annualized_cost_drag": _difference(
                top_n["gross_annualized_return"],
                top_n["net_annualized_return"],
            ),
            "top_n_is_bankrupt": _is_bankrupt(returns),
            "spread_is_bankrupt": _is_bankrupt(spread),
        }
    )
    coverage = series.get("coverage", pl.DataFrame())
    row["mean_coverage"] = (
        coverage.get_column("coverage_ratio").mean()
        if "coverage_ratio" in coverage.columns and not coverage.is_empty()
        else None
    )
    return row, {
        "summary": pl.DataFrame([row]),
        "coverage": coverage,
    }


def _ic_tables(
    series: Mapping[str, pl.DataFrame],
    selected: set[str],
    annualization: int,
    periods: pl.DataFrame | None,
) -> dict[str, pl.DataFrame]:
    ic = _ic_frame(series, periods)
    tables: dict[str, pl.DataFrame] = {}
    if {"ic_time_series", "ic_histogram"} & selected:
        tables["ic"] = ic
    if "rolling_ic_mean" in selected and not ic.is_empty():
        tables["rolling_ic_mean"] = ic.sort(TIME).select(
            TIME,
            pl.col("pearson_ic")
            .rolling_mean(window_size=annualization)
            .alias("pearson_ic"),
            pl.col("spearman_ic")
            .rolling_mean(window_size=annualization)
            .alias("spearman_ic"),
        )
    if "ic_decay" in selected:
        decay = series.get("ic_decay", pl.DataFrame())
        if decay.is_empty():
            tables["ic_decay"] = decay
        elif "ic" in decay.columns:
            tables["ic_decay"] = (
                decay.group_by("lag", "method")
                .agg(pl.col("ic").mean().alias("ic_mean"))
                .sort(["method", "lag"])
            )
        else:
            # Canonical Strategy artifacts persist the already-aggregated
            # decay table.  Reusing it must not attempt to aggregate a
            # non-existent per-date ``ic`` column again.
            tables["ic_decay"] = decay.select("lag", "method", "ic_mean").sort(
                ["method", "lag"]
            )
    return tables


def _spread_tables(
    series: Mapping[str, pl.DataFrame],
    selected: set[str],
    annualization: int,
) -> dict[str, pl.DataFrame]:
    all_returns = _lag_period_returns(series, portfolio="spread")
    lag_zero = (
        all_returns.filter(pl.col("lag") == 0)
        if not all_returns.is_empty()
        else all_returns
    )
    tables: dict[str, pl.DataFrame] = {}
    if {"spread_time_series", "spread_histogram"} & selected:
        tables["spread_returns"] = _add_grouped_cumulative(lag_zero, ())
    if "spread_rolling_vol" in selected:
        tables["spread_rolling_vol"] = _rolling_performance(lag_zero, annualization)
    if "spread_lag_performance" in selected:
        tables["spread_lag_performance"] = _lag_performance(all_returns, annualization)
    return tables


def _top_n_tables(
    returns: pl.DataFrame,
    series: Mapping[str, pl.DataFrame],
    selected: set[str],
    annualization: int,
    benchmark_returns: pl.DataFrame | None,
) -> dict[str, pl.DataFrame]:
    tables: dict[str, pl.DataFrame] = {}
    value = _add_grouped_cumulative(returns, ())
    aligned_benchmarks = (
        benchmark_returns.join(returns.select(TIME), on=TIME, how="inner")
        if benchmark_returns is not None
        else None
    )
    if "benchmark_comparison" in selected:
        tables["top_n_returns"] = value
        if aligned_benchmarks is not None:
            tables["benchmark_returns"] = _benchmark_paths(aligned_benchmarks)
            tables["benchmark_performance"] = benchmark_performance(
                aligned_benchmarks,
                annualization=annualization,
            )
    if "excess_return" in selected and aligned_benchmarks is not None:
        tables["excess_returns"] = top_n_excess_returns(
            returns.select(TIME, "gross_return", "net_return"),
            aligned_benchmarks,
        )
    if "drawdowns" in selected:
        tables["top_n_drawdown"] = _drawdowns(returns, ())
        if aligned_benchmarks is not None:
            excess = top_n_excess_returns(
                returns.select(TIME, "gross_return", "net_return"),
                aligned_benchmarks,
            )
            tables["excess_drawdown"] = _excess_drawdowns(excess)
    if "rolling_vol" in selected:
        tables["top_n_rolling_vol"] = _rolling_performance(returns, annualization)
    if "lag_performance" in selected:
        lagged = _lag_period_returns(series, portfolio="top_n")
        tables["top_n_lag_performance"] = _lag_performance(lagged, annualization)
        if aligned_benchmarks is not None:
            tables["benchmark_performance"] = benchmark_performance(
                aligned_benchmarks,
                annualization=annualization,
            )
    return tables


def _quantile_tables(
    series: Mapping[str, pl.DataFrame],
    selected: set[str],
    annualization: int,
) -> dict[str, pl.DataFrame]:
    returns = series.get("quantile_returns", pl.DataFrame())
    tables: dict[str, pl.DataFrame] = {}
    if returns.is_empty():
        return tables
    if {"annualized_return", "annualized_sharpe"} & selected:
        rows = []
        for frame in returns.partition_by("quantile", maintain_order=True):
            metrics = _single_return_metrics(frame.get_column("return"), annualization)
            bankruptcy_time = _bankruptcy_time(frame)
            rows.append(
                {
                    "quantile": frame.get_column("quantile")[0],
                    **metrics,
                    "is_bankrupt": bankruptcy_time is not None,
                    "bankruptcy_time": bankruptcy_time,
                }
            )
        tables["quantile_performance"] = sort_quantile_frame(pl.DataFrame(rows))
    if "time_series" in selected:
        tables["quantile_returns"] = sort_quantile_frame(
            returns,
            after=(TIME,),
        ).with_columns(
            ((1.0 + pl.col("return")).cum_prod().over("quantile") - 1.0).alias(
                "cumulative_return"
            )
        )
    return tables


def _statistical_tests(
    series: Mapping[str, pl.DataFrame],
    periods: pl.DataFrame | None,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    ic = _ic_frame(series, periods)
    for name, column in (("pearson_ic", "pearson_ic"), ("spearman_ic", "spearman_ic")):
        values = ic.get_column(column).drop_nulls() if column in ic.columns else []
        rows.append(_test_row(name, one_sample_t_test(values)))
    quantile_rank_ic = quantile_rank_information_coefficients(
        series.get("quantile_returns", pl.DataFrame()),
        periods=periods,
    )
    values = (
        quantile_rank_ic.get_column("quantile_rank_ic").drop_nulls()
        if "quantile_rank_ic" in quantile_rank_ic.columns
        else []
    )
    rows.append(_test_row("quantile_rank_ic", one_sample_t_test(values)))
    spread = _lag_period_returns(series, portfolio="spread", lag=0)
    policy_returns = _complete_policy_returns(
        spread, series.get("factor", pl.DataFrame())
    )
    values = (
        policy_returns.get_column("net_return").drop_nulls()
        if "net_return" in policy_returns.columns
        else []
    )
    rows.append(_test_row("spread_net_return", one_sample_t_test(values)))
    factor_returns = _filter_to_period_starts(
        series.get("factor_returns", pl.DataFrame()), periods
    )
    values = (
        factor_returns.get_column("lambda_return").drop_nulls()
        if "lambda_return" in factor_returns.columns
        else []
    )
    rows.append(_test_row("cross_section_regression", one_sample_t_test(values)))
    return pl.DataFrame(rows)


def _test_row(name: str, result: object) -> dict[str, object]:
    return {
        "test": name,
        "sample_mean": result.mean,
        "t_value": result.t_value,
        "p_value": result.p_value,
        "sample_size": result.sample_size,
        "reason": result.reason,
    }


def _complete_policy_returns(
    returns: pl.DataFrame, factor: pl.DataFrame
) -> pl.DataFrame:
    if returns.is_empty() or factor.is_empty():
        return pl.DataFrame(
            schema={TIME: pl.Date, "gross_return": pl.Float64, "net_return": pl.Float64}
        )
    schedule = (
        factor.select(TIME)
        .unique()
        .sort(TIME)
        .with_columns(pl.col(TIME).shift(-1).alias("_next_signal"))
        .drop_nulls("_next_signal")
    )
    if schedule.is_empty():
        return pl.DataFrame(
            schema={TIME: pl.Date, "gross_return": pl.Float64, "net_return": pl.Float64}
        )
    return (
        returns.sort(TIME)
        .join_asof(schedule, on=TIME, strategy="backward")
        .filter(pl.col(TIME) < pl.col("_next_signal"))
        .group_by("_next_signal")
        .agg(
            ((1.0 + pl.col("gross_return")).product() - 1.0).alias("gross_return"),
            ((1.0 + pl.col("net_return")).product() - 1.0).alias("net_return"),
        )
        .rename({"_next_signal": TIME})
        .sort(TIME)
    )


def _lag_period_returns(
    series: Mapping[str, pl.DataFrame],
    *,
    portfolio: str,
    lag: int | None = None,
) -> pl.DataFrame:
    frame = series.get("lag_returns", pl.DataFrame())
    if frame.is_empty() or not {"gross_return", "net_return"}.issubset(frame.columns):
        return pl.DataFrame(
            schema={
                "lag": pl.Int64,
                "portfolio": pl.String,
                TIME: pl.Date,
                "gross_return": pl.Float64,
                "net_return": pl.Float64,
                "is_bankrupt": pl.Boolean,
                "bankruptcy_event": pl.Boolean,
            }
        )
    filtered = frame.filter(pl.col("portfolio") == portfolio)
    if lag is not None:
        filtered = filtered.filter(pl.col("lag") == lag)
    if "is_bankrupt" not in filtered.columns:
        filtered = filtered.with_columns(pl.lit(False).alias("is_bankrupt"))
    if "bankruptcy_event" not in filtered.columns:
        filtered = filtered.with_columns(pl.lit(False).alias("bankruptcy_event"))
    return filtered.select(
        "lag",
        "portfolio",
        TIME,
        "gross_return",
        "net_return",
        "is_bankrupt",
        "bankruptcy_event",
    ).sort(["lag", TIME])


def _paired_return_metrics(
    frame: pl.DataFrame, annualization: int
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for prefix, column in (("gross", "gross_return"), ("net", "net_return")):
        metrics = _single_return_metrics(
            frame.get_column(column) if column in frame.columns else [],
            annualization,
        )
        for name, value in metrics.items():
            result[f"{prefix}_{name}"] = value
    return result


def _single_return_metrics(
    values: object, annualization: int
) -> dict[str, float | None]:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ]
    )
    if finite.size == 0:
        return {
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "calmar": None,
        }
    total = float(np.prod(1.0 + finite) - 1.0)
    raw_annualized_return = _annualized_return(
        1.0 + total,
        periods=int(finite.size),
        annualization=annualization,
    )
    annualized_return = (
        raw_annualized_return if math.isfinite(raw_annualized_return) else None
    )
    std = float(finite.std(ddof=1)) if finite.size > 1 else math.nan
    annualized_volatility = (
        float(std * math.sqrt(annualization)) if math.isfinite(std) else None
    )
    sharpe = (
        float(finite.mean() / std * math.sqrt(annualization))
        if std != 0 and math.isfinite(std)
        else None
    )
    wealth = np.cumprod(1.0 + finite)
    peaks = np.maximum.accumulate(np.maximum(wealth, 1.0))
    max_drawdown = float(np.min(wealth / peaks - 1.0))
    calmar = (
        float(annualized_return / abs(max_drawdown))
        if max_drawdown < 0 and annualized_return is not None
        else None
    )
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def _annualized_turnover(turnover: pl.DataFrame, annualization: int) -> float | None:
    if turnover.is_empty() or "turnover" not in turnover.columns:
        return None
    values = turnover.get_column("turnover").drop_nulls()
    values = values.filter(values.is_finite())
    mean = values.mean()
    return (
        float(mean) * annualization
        if mean is not None and math.isfinite(float(mean))
        else None
    )


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _annualized_relative_wealth_excess(
    returns: pl.DataFrame,
    benchmark_returns: pl.DataFrame | None,
    annualization: int,
) -> float | None:
    """Annualize the relative wealth of net returns over one benchmark."""

    if (
        returns.is_empty()
        or benchmark_returns is None
        or benchmark_returns.is_empty()
        or not {TIME, "net_return"}.issubset(returns.columns)
        or not {TIME, "benchmark", "return"}.issubset(benchmark_returns.columns)
        or benchmark_returns.get_column("benchmark").n_unique() != 1
    ):
        return None
    aligned = returns.select(TIME, "net_return").join(
        benchmark_returns.select(TIME, "return"),
        on=TIME,
        how="inner",
    )
    if aligned.height != returns.height:
        return None
    portfolio = np.asarray(aligned.get_column("net_return"), dtype=float)
    benchmark = np.asarray(aligned.get_column("return"), dtype=float)
    if (
        not np.isfinite(portfolio).all()
        or not np.isfinite(benchmark).all()
        or np.any(portfolio < -1.0)
        or np.any(benchmark <= -1.0)
    ):
        return None
    portfolio_wealth = float(np.prod(1.0 + portfolio))
    benchmark_wealth = float(np.prod(1.0 + benchmark))
    if benchmark_wealth <= 0.0 or portfolio_wealth < 0.0:
        return None
    return float(
        (portfolio_wealth / benchmark_wealth) ** (annualization / aligned.height) - 1.0
    )


def _ic_summary(values: object, annualization: int) -> dict[str, float | None]:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ]
    )
    if finite.size == 0:
        return {"mean": None, "icir": None}
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else math.nan
    return {
        "mean": mean,
        "icir": mean / std * math.sqrt(annualization)
        if std != 0 and math.isfinite(std)
        else None,
    }


def _ic_frame(
    series: Mapping[str, pl.DataFrame],
    periods: pl.DataFrame | None = None,
) -> pl.DataFrame:
    frame = series.get("ic", pl.DataFrame())
    if "spearman_ic" not in frame.columns and "ic" in frame.columns:
        frame = frame.with_columns(pl.col("ic").alias("spearman_ic"))
    return _filter_to_period_starts(frame, periods)


def _filter_to_period_starts(
    frame: pl.DataFrame,
    periods: pl.DataFrame | None,
) -> pl.DataFrame:
    if periods is None or frame.is_empty() or TIME not in frame.columns:
        return frame
    return frame.join(periods.select(TIME), on=TIME, how="inner").sort(TIME)


def _add_grouped_cumulative(
    frame: pl.DataFrame, groups: tuple[str, ...]
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    ordered = frame.sort([*groups, TIME] if groups else [TIME])
    expressions = []
    for column in ("gross_return", "net_return"):
        cumulative = (1.0 + pl.col(column)).cum_prod()
        if groups:
            cumulative = cumulative.over(*groups)
        expressions.append((cumulative - 1.0).alias(f"{column}_cumulative"))
    return ordered.with_columns(*expressions)


def _drawdowns(frame: pl.DataFrame, groups: tuple[str, ...]) -> pl.DataFrame:
    value = _add_grouped_cumulative(frame, groups)
    expressions = []
    for column in ("gross_return", "net_return"):
        wealth = 1.0 + pl.col(f"{column}_cumulative")
        peak = wealth.cum_max()
        if groups:
            peak = peak.over(*groups)
        peak = pl.max_horizontal(peak, pl.lit(1.0))
        expressions.append((wealth / peak - 1.0).alias(f"{column}_drawdown"))
    return value.with_columns(*expressions)


def _rolling_performance(frame: pl.DataFrame, annualization: int) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return rolling_performance(
        frame.select(TIME, "gross_return", "net_return"),
        annualization=annualization,
    )


def _lag_performance(frame: pl.DataFrame, annualization: int) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    rows = []
    for part in frame.partition_by("lag", maintain_order=True):
        row: dict[str, object] = {"lag": part.get_column("lag")[0]}
        for prefix, column in (("gross", "gross_return"), ("net", "net_return")):
            metrics = _single_return_metrics(part.get_column(column), annualization)
            row[f"{prefix}_annualized_return"] = metrics["annualized_return"]
            row[f"{prefix}_sharpe"] = metrics["sharpe"]
        bankruptcy_time = _bankruptcy_time(part)
        row["is_bankrupt"] = bankruptcy_time is not None
        row["bankruptcy_time"] = bankruptcy_time
        rows.append(row)
    return pl.DataFrame(rows).sort("lag")


def _is_bankrupt(frame: pl.DataFrame) -> bool:
    return bool(
        not frame.is_empty()
        and "is_bankrupt" in frame.columns
        and frame.get_column("is_bankrupt").any()
    )


def _bankruptcy_time(frame: pl.DataFrame) -> object:
    if frame.is_empty() or "bankruptcy_event" not in frame.columns:
        return None
    return frame.filter(pl.col("bankruptcy_event")).get_column(TIME).min()


def _bankruptcies(
    *,
    returns: pl.DataFrame | None = None,
    lag_returns: pl.DataFrame | None = None,
    quantile_returns: pl.DataFrame | None = None,
) -> pl.DataFrame:
    schema = {
        "portfolio": pl.String,
        "lag": pl.Int64,
        "quantile": pl.String,
        "bankruptcy_time": pl.Date,
    }
    frames: list[pl.DataFrame] = []
    if (
        returns is not None
        and not returns.is_empty()
        and "bankruptcy_event" in returns.columns
    ):
        events = returns.filter(pl.col("bankruptcy_event"))
        if not events.is_empty():
            frames.append(
                events.select(
                    pl.lit("top_n").alias("portfolio"),
                    pl.lit(None, dtype=pl.Int64).alias("lag"),
                    pl.lit(None, dtype=pl.String).alias("quantile"),
                    pl.col(TIME).alias("bankruptcy_time"),
                )
            )
    if (
        lag_returns is not None
        and not lag_returns.is_empty()
        and "bankruptcy_event" in lag_returns.columns
    ):
        events = lag_returns.filter(pl.col("bankruptcy_event"))
        if not events.is_empty():
            frames.append(
                events.select(
                    "portfolio",
                    "lag",
                    pl.lit(None, dtype=pl.String).alias("quantile"),
                    pl.col(TIME).alias("bankruptcy_time"),
                )
            )
    if (
        quantile_returns is not None
        and not quantile_returns.is_empty()
        and "bankruptcy_event" in quantile_returns.columns
    ):
        events = quantile_returns.filter(pl.col("bankruptcy_event"))
        if not events.is_empty():
            frames.append(
                events.select(
                    pl.lit("quantile").alias("portfolio"),
                    pl.lit(None, dtype=pl.Int64).alias("lag"),
                    "quantile",
                    pl.col(TIME).alias("bankruptcy_time"),
                )
            )
    if not frames:
        return pl.DataFrame(schema=schema)
    return (
        pl.concat(frames)
        .unique()
        .sort(
            ["portfolio", "lag", "quantile", "bankruptcy_time"],
            nulls_last=True,
        )
    )


def _benchmark_paths(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(["benchmark", TIME]).with_columns(
        ((1.0 + pl.col("return")).cum_prod().over("benchmark") - 1.0).alias(
            "cumulative_return"
        )
    )


def _excess_drawdowns(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = frame.sort(["benchmark", "portfolio", TIME])
    wealth = 1.0 + pl.col("relative_wealth_excess_return")
    return ordered.with_columns(
        (
            wealth
            / pl.max_horizontal(
                wealth.cum_max().over("benchmark", "portfolio"),
                pl.lit(1.0),
            )
            - 1.0
        ).alias("drawdown")
    )


__all__ = ["compute_window_tables"]
