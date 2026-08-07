"""Fast date-window aggregation over cached factor and portfolio primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import polars as pl

from .benchmarks import benchmark_performance, top_n_excess_returns
from .inputs import TIME
from .performance import rolling_performance
from .statistics import one_sample_t_test


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
    benchmark_returns: pl.DataFrame | None = None,
) -> tuple[dict[str, bool | float | int | None], dict[str, pl.DataFrame]]:
    """Build one result section from already-windowed primitive series."""

    selected = set(items)
    if section == "summary":
        metrics, tables = _summary(
            returns,
            turnover,
            series,
            annualization,
            ic_annualization,
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
        return {}, _ic_tables(series, selected, ic_annualization)
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
            "statistical_tests": _statistical_tests(series),
            "bankruptcies": _bankruptcies(
                lag_returns=lag_returns.filter(
                    (pl.col("portfolio") == "spread") & (pl.col("lag") == 0)
                )
                if not lag_returns.is_empty()
                else pl.DataFrame()
            ),
        }
    raise ValueError(f"unknown result section: {section}")


def _summary(
    returns: pl.DataFrame,
    turnover: pl.DataFrame,
    series: Mapping[str, pl.DataFrame],
    annualization: int,
    ic_annualization: int,
) -> tuple[dict[str, bool | float | int | None], dict[str, pl.DataFrame]]:
    top_n = _paired_return_metrics(returns, annualization)
    spread = _lag_period_returns(series, portfolio="spread", lag=0)
    spread_metrics = _paired_return_metrics(spread, annualization)
    ic = _ic_frame(series)
    row: dict[str, bool | float | int | None] = {}
    for method, column in (("pearson", "pearson_ic"), ("spearman", "spearman_ic")):
        values = ic.get_column(column).drop_nulls() if column in ic.columns else []
        summary = _ic_summary(values, ic_annualization)
        row[f"{method}_ic"] = summary["mean"]
        row[f"{method}_icir"] = summary["icir"]
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
            "top_n_net_sharpe": top_n["net_sharpe"],
            "top_n_net_annualized_volatility": top_n[
                "net_annualized_volatility"
            ],
            "top_n_net_max_drawdown": top_n["net_max_drawdown"],
            "top_n_net_calmar": top_n["net_calmar"],
            "top_n_annualized_turnover": _annualized_turnover(
                turnover, annualization
            ),
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
) -> dict[str, pl.DataFrame]:
    ic = _ic_frame(series)
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
        tables["ic_decay"] = (
            decay.group_by("lag", "method")
            .agg(pl.col("ic").mean().alias("ic_mean"))
            .sort(["method", "lag"])
            if not decay.is_empty()
            else decay
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
        tables["quantile_performance"] = pl.DataFrame(rows)
    if "time_series" in selected:
        tables["quantile_returns"] = returns.sort(["quantile", TIME]).with_columns(
            ((1.0 + pl.col("return")).cum_prod().over("quantile") - 1.0).alias(
                "cumulative_return"
            )
        )
    return tables


def _statistical_tests(series: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    ic = _ic_frame(series)
    for name, column in (("pearson_ic", "pearson_ic"), ("spearman_ic", "spearman_ic")):
        values = ic.get_column(column).drop_nulls() if column in ic.columns else []
        rows.append(_test_row(name, one_sample_t_test(values)))
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
    factor_returns = series.get("factor_returns", pl.DataFrame())
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
    annualized_return = float((1.0 + total) ** (annualization / finite.size) - 1.0)
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
        if max_drawdown < 0 and math.isfinite(annualized_return)
        else None
    )
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def _annualized_turnover(
    turnover: pl.DataFrame, annualization: int
) -> float | None:
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


def _ic_frame(series: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    frame = series.get("ic", pl.DataFrame())
    if "spearman_ic" not in frame.columns and "ic" in frame.columns:
        frame = frame.with_columns(pl.col("ic").alias("spearman_ic"))
    return frame


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
    return pl.concat(frames).unique().sort(
        ["portfolio", "lag", "quantile", "bankruptcy_time"],
        nulls_last=True,
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
