"""Benchmark construction and TOP N excess-return analytics."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME, validate_universe
from .performance import _annualized_return

DEFAULT_BENCHMARK = "universe_equal_weight"


def build_universe_benchmark_returns(
    forward_returns: pl.DataFrame,
    *,
    universe: pl.DataFrame | None = None,
    sizes: pl.DataFrame | None = None,
    name: str = DEFAULT_BENCHMARK,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build an equal- or size-weighted benchmark over point-in-time membership.

    ``sizes`` must contain ``time``, ``asset_id``, and ``size``. Missing or
    non-positive sizes are excluded and the available sample is renormalized.
    """

    if not name.strip():
        raise InputValidationError("benchmark name must not be blank")
    required = {TIME, ASSET_ID, "forward_return"}
    missing = sorted(required - set(forward_returns.columns))
    if missing:
        raise InputValidationError(
            f"forward_returns is missing required columns: {missing}"
        )
    returns = forward_returns.select(TIME, ASSET_ID, "forward_return").with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("forward_return").cast(pl.Float64, strict=False),
    )
    if universe is None and sizes is None:
        valid = pl.col("forward_return").is_not_null() & pl.col(
            "forward_return"
        ).is_finite()
        aggregated = (
            returns.group_by(TIME)
            .agg(
                pl.len().cast(pl.Int64).alias("expected_count"),
                valid.sum().cast(pl.Int64).alias("observed_count"),
                pl.col("forward_return").filter(valid).mean().alias("return"),
            )
            .sort(TIME)
        )
        benchmark = aggregated.filter(pl.col("observed_count") > 0).select(
            TIME, pl.lit(name).alias("benchmark"), "return"
        )
        coverage = aggregated.select(
            TIME,
            pl.lit(name).alias("benchmark"),
            "expected_count",
            "observed_count",
            (
                pl.col("observed_count") / pl.col("expected_count")
            ).alias("coverage_ratio"),
        )
        return benchmark, coverage
    members = (
        validate_universe(universe)
        if universe is not None
        else returns.select(TIME, ASSET_ID).unique().sort([TIME, ASSET_ID])
    )
    # A return labelled at ``time`` requires a later valuation observation.
    # Universe membership can legitimately extend through the final session,
    # which has no forward interval and therefore must not be reported as a
    # zero-coverage benchmark sample.
    members = members.join(
        returns.select(TIME).unique(),
        on=TIME,
        how="inner",
    )
    expected = members.group_by(TIME).agg(
        pl.len().cast(pl.Int64).alias("expected_count")
    )
    available = members.join(returns, on=[TIME, ASSET_ID], how="inner").filter(
        pl.col("forward_return").is_not_null()
        & pl.col("forward_return").is_finite()
    )
    if sizes is None:
        weighted = available.with_columns(pl.lit(1.0).alias("_weight"))
    else:
        size_required = {TIME, ASSET_ID, "size"}
        size_missing = sorted(size_required - set(sizes.columns))
        if size_missing:
            raise InputValidationError(
                f"sizes is missing required columns: {size_missing}"
            )
        normalized_sizes = sizes.select(TIME, ASSET_ID, "size").with_columns(
            pl.col(TIME).cast(pl.Date, strict=False),
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("size").cast(pl.Float64, strict=False),
        )
        if normalized_sizes.select(
            pl.struct(TIME, ASSET_ID).is_duplicated().any()
        ).item():
            raise InputValidationError("sizes must be unique by (time, asset_id)")
        weighted = available.join(
            normalized_sizes, on=[TIME, ASSET_ID], how="inner"
        ).filter(
            pl.col("size").is_not_null()
            & pl.col("size").is_finite()
            & (pl.col("size") > 0)
        ).rename({"size": "_weight"})
    observed = weighted.group_by(TIME).agg(
        pl.len().cast(pl.Int64).alias("observed_count")
    )
    benchmark = (
        weighted.group_by(TIME)
        .agg(
            (
                (pl.col("forward_return") * pl.col("_weight")).sum()
                / pl.col("_weight").sum()
            ).alias("return")
        )
        .with_columns(pl.lit(name).alias("benchmark"))
        .select(TIME, "benchmark", "return")
        .sort(TIME)
    )
    coverage = (
        expected.join(observed, on=TIME, how="left")
        .with_columns(pl.col("observed_count").fill_null(0))
        .with_columns(
            (
                pl.col("observed_count") / pl.col("expected_count")
            ).alias("coverage_ratio"),
            pl.lit(name).alias("benchmark"),
        )
        .select(
            TIME,
            "benchmark",
            "expected_count",
            "observed_count",
            "coverage_ratio",
        )
        .sort(TIME)
    )
    return benchmark, coverage


def validate_benchmark_returns(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate named benchmark daily returns."""

    if not isinstance(frame, pl.DataFrame):
        raise InputValidationError("benchmark_returns must be a polars DataFrame")
    required = {TIME, "benchmark", "return"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError(
            f"benchmark_returns is missing required columns: {missing}"
        )
    normalized = frame.select(TIME, "benchmark", "return").with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col("benchmark").cast(pl.String),
        pl.col("return").cast(pl.Float64, strict=False),
    ).drop_nulls([TIME, "benchmark", "return"])
    normalized = normalized.filter(pl.col("return").is_finite())
    if normalized.filter(pl.col("benchmark").str.strip_chars() == "").height:
        raise InputValidationError("benchmark names must not be blank")
    if normalized.select(pl.struct(TIME, "benchmark").is_duplicated().any()).item():
        raise InputValidationError(
            "benchmark_returns must be unique by (time, benchmark)"
        )
    return normalized.sort(["benchmark", TIME])


def benchmark_performance(
    returns: pl.DataFrame, *, annualization: int
) -> pl.DataFrame:
    """Summarize each cost-free benchmark return path."""

    rows: list[dict[str, object]] = []
    for frame in returns.partition_by("benchmark", maintain_order=True):
        name = str(frame.get_column("benchmark")[0])
        values = np.array(frame.get_column("return"), dtype=float)
        periods = len(values)
        total = float(np.prod(1.0 + values) - 1.0)
        mean = float(np.mean(values)) if periods else math.nan
        std = float(np.std(values, ddof=1)) if periods > 1 else math.nan
        wealth = np.cumprod(1.0 + values)
        drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
        rows.append(
            {
                "benchmark": name,
                "total_return": total,
                "annualized_return": _annualized_return(
                    1.0 + total,
                    periods=periods,
                    annualization=annualization,
                ),
                "annualized_volatility": std * math.sqrt(annualization),
                "sharpe": (
                    mean / std * math.sqrt(annualization)
                    if std != 0 and not math.isnan(std)
                    else math.nan
                ),
                "max_drawdown": (
                    float(np.min(drawdown)) if periods else math.nan
                ),
            }
        )
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "benchmark": pl.String,
                "total_return": pl.Float64,
                "annualized_return": pl.Float64,
                "annualized_volatility": pl.Float64,
                "sharpe": pl.Float64,
                "max_drawdown": pl.Float64,
            }
        )
    )


def top_n_excess_returns(
    top_n_returns: pl.DataFrame, benchmark_returns: pl.DataFrame
) -> pl.DataFrame:
    """Compare gross and net TOP N returns with each named benchmark."""

    joined = benchmark_returns.join(top_n_returns, on=TIME, how="inner").sort(
        ["benchmark", TIME]
    )
    frames: list[pl.DataFrame] = []
    for portfolio, column in (
        ("gross", "gross_return"),
        ("net", "net_return"),
    ):
        frames.append(
            joined.with_columns(
                pl.lit(portfolio).alias("portfolio"),
                (pl.col(column) - pl.col("return")).alias("daily_excess_return"),
            )
            .with_columns(
                (
                    (1.0 + pl.col("daily_excess_return"))
                    .cum_prod()
                    .over("benchmark")
                    - 1.0
                ).alias("compounded_excess_return"),
                (
                    (1.0 + pl.col(column)).cum_prod().over("benchmark")
                    / (1.0 + pl.col("return")).cum_prod().over("benchmark")
                    - 1.0
                ).alias("relative_wealth_excess_return"),
            )
            .select(
                TIME,
                "benchmark",
                "portfolio",
                pl.col(column).alias("portfolio_return"),
                pl.col("return").alias("benchmark_return"),
                "daily_excess_return",
                "compounded_excess_return",
                "relative_wealth_excess_return",
            )
        )
    return pl.concat(frames).sort(["benchmark", "portfolio", TIME])
