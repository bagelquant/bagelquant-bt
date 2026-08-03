"""Reusable statistical diagnostics for factor research."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from .inputs import ASSET_ID, TIME


@dataclass(frozen=True, slots=True)
class OneSampleTest:
    """Result of a two-sided one-sample Student t-test."""

    mean: float | None
    t_value: float | None
    p_value: float | None
    sample_size: int
    reason: str | None = None


def one_sample_t_test(values: object, *, null_mean: float = 0.0) -> OneSampleTest:
    """Test finite values against ``null_mean`` with an exact Student-t CDF."""

    observations: list[float] = []
    for value in values:
        if value is None:
            continue
        observation = float(value)
        if math.isfinite(observation):
            observations.append(observation)
    finite = np.asarray(observations, dtype=float)
    sample_size = int(finite.size)
    mean = float(finite.mean()) if sample_size else None
    if sample_size < 2:
        return OneSampleTest(
            mean,
            None,
            None,
            sample_size,
            "at least two samples required",
        )
    standard_deviation = float(finite.std(ddof=1))
    if standard_deviation == 0.0:
        return OneSampleTest(mean, None, None, sample_size, "sample variance is zero")
    result = stats.ttest_1samp(finite, popmean=null_mean, alternative="two-sided")
    return OneSampleTest(
        mean,
        float(result.statistic),
        float(result.pvalue),
        sample_size,
    )


def cross_sectional_factor_returns(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Estimate per-date OLS factor-return slopes with an intercept."""

    paired = (
        factor.select(TIME, ASSET_ID, "factor")
        .join(
            forward_returns.select(TIME, ASSET_ID, "forward_return"),
            on=[TIME, ASSET_ID],
            how="inner",
        )
        .drop_nulls(["factor", "forward_return"])
    )
    if paired.is_empty():
        return pl.DataFrame(
            schema={TIME: pl.Date, "lambda_return": pl.Float64, "sample_size": pl.Int64}
        )
    return (
        paired.group_by(TIME)
        .agg(
            pl.len().alias("sample_size"),
            pl.col("factor").n_unique().alias("_factor_values"),
            pl.col("factor").var().alias("_factor_variance"),
            pl.cov("factor", "forward_return").alias("_covariance"),
        )
        .with_columns(
            pl.when(
                (pl.col("sample_size") >= 3)
                & (pl.col("_factor_values") >= 2)
                & (pl.col("_factor_variance") > 0)
            )
            .then(pl.col("_covariance") / pl.col("_factor_variance"))
            .otherwise(None)
            .alias("lambda_return")
        )
        .select(TIME, "lambda_return", "sample_size")
        .sort(TIME)
    )


__all__ = ["OneSampleTest", "cross_sectional_factor_returns", "one_sample_t_test"]
