"""Reusable statistical diagnostics for factor research."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from bagelquant_core import quantile_rank_information_coefficient
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


def quantile_rank_information_coefficients(
    quantile_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Compute per-period rank IC from complete q1-to-qN gross returns."""

    schema = {
        TIME: pl.Date,
        "quantile_rank_ic": pl.Float64,
        "quantiles": pl.Int64,
    }
    if quantile_returns.is_empty():
        return pl.DataFrame(schema=schema)
    required = {TIME, "quantile", "return"}
    missing = required - set(quantile_returns.columns)
    if missing:
        raise ValueError(
            f"quantile returns are missing required columns: {sorted(missing)}"
        )
    labels = quantile_returns.get_column("quantile").drop_nulls().unique().to_list()
    numbers: list[int] = []
    for label in labels:
        text = str(label)
        if not text.startswith("q") or not text[1:].isdigit() or int(text[1:]) < 1:
            raise ValueError(f"invalid quantile label: {text}")
        numbers.append(int(text[1:]))
    quantiles = max(numbers, default=0)
    if quantiles < 2 or set(numbers) != set(range(1, quantiles + 1)):
        raise ValueError("quantile returns require contiguous q1-to-qN labels")
    expected_labels = [f"q{number}" for number in range(1, quantiles + 1)]
    rows: list[dict[str, object]] = []
    for period in quantile_returns.get_column(TIME).unique().sort().to_list():
        sample = quantile_returns.filter(pl.col(TIME) == period)
        if sample.get_column("quantile").n_unique() != sample.height:
            raise ValueError(f"duplicate quantile returns for {period}")
        by_label = {
            str(row["quantile"]): row["return"]
            for row in sample.select("quantile", "return").iter_rows(named=True)
        }
        result = (
            quantile_rank_information_coefficient(
                range(quantiles, 0, -1),
                [by_label.get(label) for label in expected_labels],
                quantiles=quantiles,
            )
            if set(by_label) == set(expected_labels)
            else None
        )
        rows.append(
            {
                TIME: period,
                "quantile_rank_ic": result,
                "quantiles": quantiles,
            }
        )
    return pl.DataFrame(rows, schema=schema).sort(TIME)


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


__all__ = [
    "OneSampleTest",
    "cross_sectional_factor_returns",
    "one_sample_t_test",
    "quantile_rank_information_coefficients",
]
