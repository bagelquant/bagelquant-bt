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
from .returns import align_signal_to_forward_returns

FACTOR_LAGS = (0, 1, 2, 3, 4, 5, 10, 20, 30, 60)


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
    factor, forward_returns = align_signal_to_forward_returns(
        aligned_factor,
        aligned_prices,
        value_column="factor",
        label="factor",
    )
    if factor.is_empty():
        raise InputValidationError("at least two overlapping price times are required")

    ic = information_coefficients(factor, forward_returns)
    ic_summary = summarize_ic(ic)
    values = np.array(ic["spearman_ic"].drop_nulls(), dtype=float)
    ic_std = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
    ic_mean = float(np.mean(values)) if len(values) else math.nan
    icir = ic_mean / ic_std if ic_std != 0 and not math.isnan(ic_std) else math.nan
    quantile_returns = traded_factor_quantile_returns(
        factor,
        aligned_prices,
        config=config,
        quantiles=config.quantiles,
    )
    top_minus_bottom = _top_minus_bottom(quantile_returns, config.quantiles)
    top_n_weights = top_n_equal_weights(factor, top_n=config.top_n)
    top_n_backtest = backtest_weight_frame(
        top_n_weights,
        aligned_prices,
        config=config,
    )
    long_short_weights = long_short_quantile_weights(
        factor,
        quantiles=config.quantiles,
    )
    long_short_backtest = (
        backtest_weight_frame(
            long_short_weights,
            aligned_prices,
            config=config,
        )
        if long_short_weights.height
        else None
    )
    lag_analysis = factor_lag_analysis(
        factor,
        aligned_prices,
        config=config,
        lags=FACTOR_LAGS,
    )
    lag_returns = factor_lag_returns(
        factor,
        aligned_prices,
        config=config,
        lags=FACTOR_LAGS,
    )
    ic_decay = factor_ic_decay(
        factor,
        forward_returns,
        lags=FACTOR_LAGS,
    )

    return FactorEvaluationResult(
        factor=factor,
        forward_returns=forward_returns,
        ic=ic,
        ic_summary=ic_summary,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        quantile_returns=quantile_returns,
        top_minus_bottom=top_minus_bottom,
        top_n_weights=top_n_weights,
        top_n_backtest=top_n_backtest,
        long_short_weights=long_short_weights,
        long_short_backtest=long_short_backtest,
        lag_analysis=lag_analysis,
        lag_returns=lag_returns,
        ic_decay=ic_decay,
    )


def information_coefficients(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Compute daily Pearson and Spearman cross-sectional IC."""

    pearson = _information_coefficient_values(
        factor,
        forward_returns,
        method="pearson",
        output_column="pearson_ic",
    )
    spearman = _information_coefficient_values(
        factor,
        forward_returns,
        method="spearman",
        output_column="spearman_ic",
    )
    return pearson.join(spearman, on=TIME, how="full", coalesce=True).sort(TIME)


def information_coefficient(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    method: str = "spearman",
) -> pl.DataFrame:
    """Compute daily cross-sectional IC."""

    return _information_coefficient_values(
        factor,
        forward_returns,
        method=method,
        output_column="ic",
    )


def summarize_ic(ic: pl.DataFrame) -> pl.DataFrame:
    """Summarize Pearson and Spearman IC series."""

    rows: list[dict[str, object]] = []
    for method, column in (("pearson", "pearson_ic"), ("spearman", "spearman_ic")):
        values = np.array(ic[column].drop_nulls(), dtype=float)
        std = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        mean = float(np.mean(values)) if len(values) else math.nan
        icir = mean / std if std != 0 and not math.isnan(std) else math.nan
        rows.append({"method": method, "mean": mean, "std": std, "icir": icir})
    return pl.DataFrame(rows)


def _information_coefficient_values(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    method: str,
    output_column: str,
) -> pl.DataFrame:
    """Compute daily cross-sectional IC for one method."""

    paired = factor.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
    rows: list[dict[str, object]] = []
    for time, group in paired.partition_by(TIME, as_dict=True).items():
        values = group.drop_nulls(["factor", "forward_return"])
        if values.height < 2:
            rows.append({TIME: _partition_key(time), output_column: None})
            continue
        left = np.array(values["factor"], dtype=float)
        right = np.array(values["forward_return"], dtype=float)
        if method == "spearman":
            left = _average_rank(left)
            right = _average_rank(right)
        if len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
            rows.append({TIME: _partition_key(time), output_column: None})
            continue
        rows.append(
            {
                TIME: _partition_key(time),
                output_column: float(np.corrcoef(left, right)[0, 1]),
            }
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


def traded_factor_quantile_returns(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    quantiles: int,
) -> pl.DataFrame:
    """Compute quantile return series by trading quantile portfolios."""

    rows: list[dict[str, object]] = []
    for quantile, weights in quantile_equal_weights(
        factor,
        quantiles=quantiles,
    ).items():
        if weights.is_empty():
            continue
        backtest = backtest_weight_frame(weights, prices, config=config)
        for row in backtest.returns.iter_rows(named=True):
            rows.append(
                {
                    TIME: row[TIME],
                    "quantile": quantile,
                    "return": row["gross_return"],
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "quantile": pl.String,
                "return": pl.Float64,
                "cumulative_return": pl.Float64,
            }
        )
    returns = pl.DataFrame(rows).sort([TIME, "quantile"])
    return returns.with_columns(
        (
            (1.0 + pl.col("return").fill_null(0.0)).cum_prod().over("quantile") - 1.0
        ).alias("cumulative_return")
    )


def quantile_equal_weights(
    factor: pl.DataFrame,
    *,
    quantiles: int,
) -> dict[str, pl.DataFrame]:
    """Convert factor scores into equal-weight portfolios by quantile."""

    rows: dict[str, list[dict[str, object]]] = {
        f"q{number}": [] for number in range(1, quantiles + 1)
    }
    for time, group in factor.partition_by(TIME, as_dict=True).items():
        values = group.drop_nulls("factor").sort("factor")
        if values.height < quantiles:
            continue
        ranked = values.with_row_index("rank", offset=1).with_columns(
            (((pl.col("rank") - 1) * quantiles / pl.len()).floor() + 1)
            .cast(pl.Int64)
            .alias("bucket")
        )
        for bucket, bucket_group in ranked.partition_by("bucket", as_dict=True).items():
            bucket_number = int(_partition_key(bucket))
            quantile = f"q{bucket_number}"
            weight = 1.0 / bucket_group.height
            for row in bucket_group.iter_rows(named=True):
                rows[quantile].append(
                    {
                        TIME: _partition_key(time),
                        ASSET_ID: row[ASSET_ID],
                        "weight": weight,
                    }
                )

    empty = pl.DataFrame(
        schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64}
    )
    return {
        quantile: (
            pl.DataFrame(weight_rows).sort([TIME, ASSET_ID])
            if weight_rows
            else empty.clone()
        )
        for quantile, weight_rows in rows.items()
    }


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


def long_short_quantile_weights(
    factor: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    """Convert factor scores into top-long and bottom-short weights."""

    rows: list[dict[str, object]] = []
    for time, group in factor.partition_by(TIME, as_dict=True).items():
        values = group.drop_nulls("factor").sort("factor")
        if values.height < quantiles:
            continue
        ranked = values.with_row_index("rank", offset=1).with_columns(
            (((pl.col("rank") - 1) * quantiles / pl.len()).floor() + 1)
            .cast(pl.Int64)
            .alias("bucket")
        )
        bottom = ranked.filter(pl.col("bucket") == 1)
        top = ranked.filter(pl.col("bucket") == quantiles)
        if bottom.is_empty() or top.is_empty():
            continue
        long_weight = 1.0 / top.height
        short_weight = -1.0 / bottom.height
        for row in top.iter_rows(named=True):
            rows.append(
                {
                    TIME: _partition_key(time),
                    ASSET_ID: row[ASSET_ID],
                    "weight": long_weight,
                }
            )
        for row in bottom.iter_rows(named=True):
            rows.append(
                {
                    TIME: _partition_key(time),
                    ASSET_ID: row[ASSET_ID],
                    "weight": short_weight,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64}
        )
    return pl.DataFrame(rows).sort([TIME, ASSET_ID])


def factor_lag_analysis(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    lags: tuple[int, ...] = FACTOR_LAGS,
) -> pl.DataFrame:
    """Backtest lagged factor signals for TOP N and long-short portfolios."""

    rows: list[dict[str, object]] = []
    for lag in lags:
        lagged = lag_factor(factor, lag=lag)
        portfolio_specs = (
            ("top_n", top_n_equal_weights(lagged, top_n=config.top_n)),
            (
                "long_short",
                long_short_quantile_weights(lagged, quantiles=config.quantiles),
            ),
        )
        for portfolio, weights in portfolio_specs:
            row: dict[str, object] = {
                "lag": lag,
                "portfolio": portfolio,
                "gross_cumulative_return": math.nan,
                "net_cumulative_return": math.nan,
                "gross_sharpe": math.nan,
                "net_sharpe": math.nan,
            }
            if weights.height:
                try:
                    backtest = backtest_weight_frame(weights, prices, config=config)
                except InputValidationError:
                    backtest = None
                if backtest is not None:
                    row.update(
                        {
                            "gross_cumulative_return": (
                                backtest.summary.gross_total_return
                            ),
                            "net_cumulative_return": backtest.summary.net_total_return,
                            "gross_sharpe": backtest.summary.gross_sharpe,
                            "net_sharpe": backtest.summary.net_sharpe,
                        }
                    )
            rows.append(row)
    return pl.DataFrame(rows).sort(["portfolio", "lag"])


def factor_lag_returns(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    lags: tuple[int, ...] = FACTOR_LAGS,
) -> pl.DataFrame:
    """Return cumulative lagged factor backtest time series."""

    rows: list[dict[str, object]] = []
    for lag in lags:
        lagged = lag_factor(factor, lag=lag)
        portfolio_specs = (
            ("top_n", top_n_equal_weights(lagged, top_n=config.top_n)),
            (
                "long_short",
                long_short_quantile_weights(lagged, quantiles=config.quantiles),
            ),
        )
        for portfolio, weights in portfolio_specs:
            if not weights.height:
                continue
            try:
                backtest = backtest_weight_frame(weights, prices, config=config)
            except InputValidationError:
                continue
            for row in backtest.value.iter_rows(named=True):
                rows.append(
                    {
                        "lag": lag,
                        "portfolio": portfolio,
                        TIME: row[TIME],
                        "gross_cumulative_return": row["gross_return_cumulative"],
                        "net_cumulative_return": row["net_return_cumulative"],
                        "gross_sharpe": backtest.summary.gross_sharpe,
                        "net_sharpe": backtest.summary.net_sharpe,
                    }
                )
    if not rows:
        return pl.DataFrame(
            schema={
                "lag": pl.Int64,
                "portfolio": pl.String,
                TIME: pl.Date,
                "gross_cumulative_return": pl.Float64,
                "net_cumulative_return": pl.Float64,
                "gross_sharpe": pl.Float64,
                "net_sharpe": pl.Float64,
            }
        )
    return pl.DataFrame(rows).sort(["portfolio", "lag", TIME])


def factor_ic_decay(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    lags: tuple[int, ...] = FACTOR_LAGS,
) -> pl.DataFrame:
    """Compute mean Pearson and Spearman IC for lagged factor signals."""

    rows: list[dict[str, object]] = []
    for lag in lags:
        lagged = lag_factor(factor, lag=lag)
        if lagged.is_empty():
            rows.extend(
                [
                    {"lag": lag, "method": "pearson", "ic_mean": math.nan},
                    {"lag": lag, "method": "spearman", "ic_mean": math.nan},
                ]
            )
            continue
        ic = information_coefficients(lagged, forward_returns)
        summary = summarize_ic(ic)
        for row in summary.iter_rows(named=True):
            rows.append(
                {
                    "lag": lag,
                    "method": row["method"],
                    "ic_mean": row["mean"],
                }
            )
    return pl.DataFrame(rows).sort(["method", "lag"])


def lag_factor(factor: pl.DataFrame, *, lag: int) -> pl.DataFrame:
    """Shift each asset's factor signal forward by lag observations."""

    if lag <= 0:
        return factor.sort([TIME, ASSET_ID])
    return (
        factor.sort([ASSET_ID, TIME])
        .with_columns(pl.col("factor").shift(lag).over(ASSET_ID).alias("factor"))
        .drop_nulls("factor")
        .sort([TIME, ASSET_ID])
    )


def _top_minus_bottom(quantile_returns: pl.DataFrame, quantiles: int) -> pl.DataFrame:
    bottom = "q1"
    top = f"q{quantiles}"
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
