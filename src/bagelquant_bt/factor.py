"""Factor evaluation routines."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .config import BacktestConfig
from .engine import _require_config, backtest_weight_frame
from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME, validate_factor, validate_prices
from .results import FactorEvaluationResult
from .returns import asset_close_to_close_returns


def run_factor_evaluation(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig | None = None,
) -> FactorEvaluationResult:
    """Evaluate a factor score frame against forward returns."""

    resolved_config = _require_config(config)
    aligned_factor = validate_factor(factor)
    aligned_prices = validate_prices(prices)
    return evaluate_factor_frame(
        aligned_factor,
        aligned_prices,
        config=resolved_config,
    )


def evaluate_factor_frame(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
) -> FactorEvaluationResult:
    """Evaluate an already materialized factor score frame."""

    aligned_factor = validate_factor(factor)
    aligned_prices = validate_prices(prices)
    forward_returns = asset_close_to_close_returns(aligned_prices)
    paired = aligned_factor.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
    if paired.is_empty():
        raise InputValidationError("at least two overlapping price times are required")

    factor = paired.select(TIME, ASSET_ID, "factor").sort([TIME, ASSET_ID])
    forward_returns = paired.select(TIME, ASSET_ID, "forward_return").sort(
        [TIME, ASSET_ID]
    )
    ic = information_coefficient(
        factor,
        forward_returns,
        method=config.ic_method,
    )
    values = np.array(ic["ic"].drop_nulls(), dtype=float)
    ic_std = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
    ic_mean = float(np.mean(values)) if len(values) else math.nan
    icir = ic_mean / ic_std if ic_std != 0 and not math.isnan(ic_std) else math.nan
    quantile_returns = factor_quantile_returns(
        factor,
        forward_returns,
        quantiles=config.quantiles,
    )
    top_minus_bottom = _top_minus_bottom(quantile_returns, config.quantiles)
    top_n_weights = top_n_equal_weights(factor, top_n=config.top_n)
    top_n_backtest = backtest_weight_frame(
        top_n_weights,
        aligned_prices,
        config=config,
    )

    return FactorEvaluationResult(
        factor=factor,
        forward_returns=forward_returns,
        ic=ic,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        quantile_returns=quantile_returns,
        top_minus_bottom=top_minus_bottom,
        top_n_weights=top_n_weights,
        top_n_backtest=top_n_backtest,
    )


def information_coefficient(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    method: str = "spearman",
) -> pl.DataFrame:
    """Compute daily cross-sectional IC."""

    paired = factor.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
    rows: list[dict[str, object]] = []
    for time, group in paired.partition_by(TIME, as_dict=True).items():
        values = group.drop_nulls(["factor", "forward_return"])
        if values.height < 2:
            rows.append({TIME: _partition_key(time), "ic": None})
            continue
        left = np.array(values["factor"], dtype=float)
        right = np.array(values["forward_return"], dtype=float)
        if method == "spearman":
            left = _average_rank(left)
            right = _average_rank(right)
        if len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
            rows.append({TIME: _partition_key(time), "ic": None})
            continue
        rows.append(
            {TIME: _partition_key(time), "ic": float(np.corrcoef(left, right)[0, 1])}
        )
    return pl.DataFrame(rows).sort(TIME)


def factor_quantile_returns(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    """Compute equal-weight daily returns by factor quantile."""

    paired = factor.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
    rows: list[dict[str, object]] = []
    for time, group in paired.partition_by(TIME, as_dict=True).items():
        values = group.drop_nulls(["factor", "forward_return"]).sort("factor")
        if values.height < quantiles:
            for number in range(1, quantiles + 1):
                rows.append(
                    {
                        TIME: _partition_key(time),
                        "quantile": f"q{number}",
                        "return": None,
                    }
                )
            continue
        ranked = values.with_row_index("rank", offset=1).with_columns(
            (((pl.col("rank") - 1) * quantiles / pl.len()).floor() + 1)
            .cast(pl.Int64)
            .alias("bucket")
        )
        grouped = ranked.group_by("bucket").agg(
            pl.col("forward_return").mean().alias("return")
        )
        by_bucket = {
            int(row["bucket"]): row["return"] for row in grouped.iter_rows(named=True)
        }
        for number in range(1, quantiles + 1):
            rows.append(
                {
                    TIME: _partition_key(time),
                    "quantile": f"q{number}",
                    "return": by_bucket.get(number),
                }
            )
    returns = pl.DataFrame(rows).sort([TIME, "quantile"])
    return returns.with_columns(
        (
            (1.0 + pl.col("return").fill_null(0.0)).cum_prod().over("quantile") - 1.0
        ).alias("cumulative_return")
    )


def top_n_equal_weights(factor: pl.DataFrame, *, top_n: int) -> pl.DataFrame:
    """Convert factor scores into long-only TOP N equal weights."""

    selected = (
        factor.drop_nulls("factor")
        .sort([TIME, "factor"], descending=[False, True])
        .with_columns(pl.int_range(1, pl.len() + 1).over(TIME).alias("rank"))
        .filter(pl.col("rank") <= top_n)
        .with_columns((1.0 / pl.len().over(TIME)).alias("weight"))
        .select(TIME, ASSET_ID, "weight")
    )
    return selected.sort([TIME, ASSET_ID])


def _top_minus_bottom(quantile_returns: pl.DataFrame, quantiles: int) -> pl.DataFrame:
    top = "q1"
    bottom = f"q{quantiles}"
    return (
        quantile_returns.filter(pl.col("quantile").is_in([bottom, top]))
        .select(TIME, "quantile", "return")
        .pivot(index=TIME, on="quantile", values="return")
        .with_columns((pl.col(top) - pl.col(bottom)).alias("top_minus_bottom"))
        .select(TIME, "top_minus_bottom")
        .sort(TIME)
    )


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _partition_key(key: object) -> object:
    if isinstance(key, tuple):
        return key[0]
    if isinstance(key, list):
        return key[0]
    return key
