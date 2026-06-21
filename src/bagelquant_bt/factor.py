"""Factor evaluation routines."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .config import BacktestConfig
from .engine import (
    _backtest_weight_frame_with_forward_returns,
    _require_config,
    backtest_weight_frame,
)
from .exceptions import InputValidationError
from .inputs import (
    ASSET_ID,
    TIME,
    asset_coverage,
    missing_price_keys,
    validate_factor,
    validate_prices,
)
from .results import BacktestResult, FactorEvaluationResult
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
    coverage = asset_coverage(
        aligned_factor,
        aligned_prices,
        asset_count_column="factor_signal_asset_count",
    )
    missing_keys = missing_price_keys(aligned_factor, aligned_prices)
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
    quantile_returns = _traded_factor_quantile_returns_with_forward_returns(
        factor,
        aligned_prices,
        config=config,
        quantiles=config.quantiles,
        forward_returns=forward_returns,
    )
    spread_returns = _spread_returns(quantile_returns, config.quantiles)
    top_n_weights = top_n_equal_weights(factor, top_n=config.top_n)
    top_n_backtest = _backtest_weight_frame_with_forward_returns(
        top_n_weights,
        aligned_prices,
        forward_returns,
        config=config,
    )
    spread_weights = spread_quantile_weights(
        factor,
        quantiles=config.quantiles,
    )
    spread_backtest = (
        _backtest_weight_frame_with_forward_returns(
            spread_weights,
            aligned_prices,
            forward_returns,
            config=config,
        )
        if spread_weights.height
        else None
    )
    lag_backtests = _lag_backtests(
        factor,
        aligned_prices,
        config=config,
        lags=FACTOR_LAGS,
        forward_returns=forward_returns,
    )
    lag_analysis = _lag_analysis_from_backtests(lag_backtests)
    lag_returns = _lag_returns_from_backtests(lag_backtests)
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
        spread_returns=spread_returns,
        top_n_weights=top_n_weights,
        top_n_backtest=top_n_backtest,
        spread_weights=spread_weights,
        spread_backtest=spread_backtest,
        lag_analysis=lag_analysis,
        lag_returns=lag_returns,
        ic_decay=ic_decay,
        coverage=coverage,
        missing_price_keys=missing_keys,
    )


def information_coefficients(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Compute daily Pearson and Spearman cross-sectional IC."""

    paired = factor.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
    if paired.is_empty():
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "pearson_ic": pl.Float64,
                "spearman_ic": pl.Float64,
            }
        )
    times = paired.select(TIME).unique()
    values = paired.drop_nulls(["factor", "forward_return"])
    if values.is_empty():
        ic = pl.DataFrame(
            schema={
                TIME: paired.schema[TIME],
                "pearson_ic": pl.Float64,
                "spearman_ic": pl.Float64,
            }
        )
    else:
        ic = (
            values.with_columns(
                pl.col("factor").rank("average").over(TIME).alias("_factor_rank"),
                pl.col("forward_return")
                .rank("average")
                .over(TIME)
                .alias("_return_rank"),
            )
            .group_by(TIME)
            .agg(
                _corr_expr("factor", "forward_return").alias("pearson_ic"),
                _corr_expr("_factor_rank", "_return_rank").alias("spearman_ic"),
            )
        )
    return times.join(ic, on=TIME, how="left").sort(TIME)


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
    if paired.is_empty():
        return pl.DataFrame(schema={TIME: pl.Date, output_column: pl.Float64})
    times = paired.select(TIME).unique()
    if method == "pearson":
        left = "factor"
        right = "forward_return"
        values = paired.drop_nulls([left, right])
    elif method == "spearman":
        left = "_factor_rank"
        right = "_return_rank"
        values = paired.drop_nulls(["factor", "forward_return"]).with_columns(
            pl.col("factor").rank("average").over(TIME).alias(left),
            pl.col("forward_return").rank("average").over(TIME).alias(right),
        )
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")
    if values.is_empty():
        ic = pl.DataFrame(schema={TIME: paired.schema[TIME], output_column: pl.Float64})
    else:
        ic = values.group_by(TIME).agg(_corr_expr(left, right).alias(output_column))
    return times.join(ic, on=TIME, how="left").sort(TIME)


def factor_quantile_returns(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    """Compute equal-weight daily returns by factor quantile."""

    paired = factor.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
    if paired.is_empty():
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "quantile": pl.String,
                "return": pl.Float64,
                "cumulative_return": pl.Float64,
            }
        )
    quantile_grid = _quantile_grid(paired.select(TIME).unique(), quantiles)
    bucket_returns = (
        _quantile_bucket_frame(
            paired.drop_nulls(["factor", "forward_return"]),
            quantiles=quantiles,
        )
        .group_by(TIME, "bucket")
        .agg(pl.col("forward_return").mean().alias("return"))
        .with_columns(
            (pl.lit("q") + pl.col("bucket").cast(pl.String)).alias("quantile")
        )
        .select(TIME, "quantile", "return")
    )
    returns = quantile_grid.join(
        bucket_returns,
        on=[TIME, "quantile"],
        how="left",
    ).sort([TIME, "quantile"])
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

    return _traded_factor_quantile_returns_with_forward_returns(
        factor,
        prices,
        config=config,
        quantiles=quantiles,
        forward_returns=None,
    )


def _traded_factor_quantile_returns_with_forward_returns(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    quantiles: int,
    forward_returns: pl.DataFrame | None,
) -> pl.DataFrame:
    """Compute traded quantile returns, optionally reusing forward returns."""

    frames: list[pl.DataFrame] = []
    for quantile, weights in quantile_equal_weights(
        factor,
        quantiles=quantiles,
    ).items():
        if weights.is_empty():
            continue
        backtest = (
            _backtest_weight_frame_with_forward_returns(
                weights,
                prices,
                forward_returns,
                config=config,
            )
            if forward_returns is not None
            else backtest_weight_frame(weights, prices, config=config)
        )
        frames.append(
            backtest.returns.select(
                TIME,
                pl.lit(quantile).alias("quantile"),
                pl.col("gross_return").alias("return"),
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "quantile": pl.String,
                "return": pl.Float64,
                "cumulative_return": pl.Float64,
            }
        )
    returns = pl.concat(frames).sort([TIME, "quantile"])
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

    empty = pl.DataFrame(
        schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64}
    )
    bucketed = _quantile_bucket_frame(factor.drop_nulls("factor"), quantiles=quantiles)
    if bucketed.is_empty():
        return {f"q{number}": empty.clone() for number in range(1, quantiles + 1)}
    weights = (
        bucketed.with_columns((1.0 / pl.len().over(TIME, "bucket")).alias("weight"))
        .select(TIME, ASSET_ID, "bucket", "weight")
        .sort([TIME, ASSET_ID])
    )
    return {
        f"q{number}": (
            weights.filter(pl.col("bucket") == number)
            .select(TIME, ASSET_ID, "weight")
            .sort([TIME, ASSET_ID])
            if weights.filter(pl.col("bucket") == number).height
            else empty.clone()
        )
        for number in range(1, quantiles + 1)
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


def spread_quantile_weights(
    factor: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    """Convert q1 and qN factor buckets into spread portfolio weights."""

    bucketed = _quantile_bucket_frame(factor.drop_nulls("factor"), quantiles=quantiles)
    selected = bucketed.filter(pl.col("bucket").is_in([1, quantiles]))
    if selected.is_empty():
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64}
        )
    return (
        selected.with_columns(
            pl.when(pl.col("bucket") == 1)
            .then(1.0 / pl.len().over(TIME, "bucket"))
            .otherwise(-1.0 / pl.len().over(TIME, "bucket"))
            .alias("weight")
        )
        .select(TIME, ASSET_ID, "weight")
        .sort([TIME, ASSET_ID])
    )


def factor_lag_analysis(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    lags: tuple[int, ...] = FACTOR_LAGS,
) -> pl.DataFrame:
    """Backtest lagged factor signals for TOP N and spread portfolios."""

    return _lag_analysis_from_backtests(
        _lag_backtests(factor, prices, config=config, lags=lags)
    )


def factor_lag_returns(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    lags: tuple[int, ...] = FACTOR_LAGS,
) -> pl.DataFrame:
    """Return cumulative lagged factor backtest time series."""

    return _lag_returns_from_backtests(
        _lag_backtests(factor, prices, config=config, lags=lags)
    )


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


def _corr_expr(left: str, right: str) -> pl.Expr:
    return (
        pl.when(
            (pl.len() < 2)
            | (pl.col(left).n_unique() < 2)
            | (pl.col(right).n_unique() < 2)
        )
        .then(None)
        .otherwise(pl.corr(left, right))
    )


def _quantile_bucket_frame(frame: pl.DataFrame, *, quantiles: int) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Int64).alias("bucket")).filter(
            pl.lit(False)
        )
    ranked = (
        frame.sort([TIME, "factor"], descending=[False, True])
        .with_columns(
            pl.len().over(TIME).alias("_count"),
            pl.int_range(1, pl.len() + 1).over(TIME).alias("_rank"),
        )
        .filter(pl.col("_count") >= quantiles)
        .with_columns(
            (((pl.col("_rank") - 1) * quantiles / pl.col("_count")).floor() + 1)
            .cast(pl.Int64)
            .alias("bucket")
        )
        .drop("_count", "_rank")
    )
    return ranked


def _quantile_grid(times: pl.DataFrame, quantiles: int) -> pl.DataFrame:
    quantile_labels = pl.DataFrame(
        {"quantile": [f"q{number}" for number in range(1, quantiles + 1)]},
        schema={"quantile": pl.String},
    )
    return times.select(TIME).unique().join(quantile_labels, how="cross")


def _lag_backtests(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    lags: tuple[int, ...],
    forward_returns: pl.DataFrame | None = None,
) -> list[tuple[int, str, BacktestResult | None]]:
    results: list[tuple[int, str, BacktestResult | None]] = []
    for lag in lags:
        lagged = lag_factor(factor, lag=lag)
        portfolio_specs = (
            ("top_n", top_n_equal_weights(lagged, top_n=config.top_n)),
            (
                "spread",
                spread_quantile_weights(lagged, quantiles=config.quantiles),
            ),
        )
        for portfolio, weights in portfolio_specs:
            backtest = None
            if weights.height:
                try:
                    backtest = (
                        _backtest_weight_frame_with_forward_returns(
                            weights,
                            prices,
                            forward_returns,
                            config=config,
                        )
                        if forward_returns is not None
                        else backtest_weight_frame(weights, prices, config=config)
                    )
                except InputValidationError:
                    backtest = None
            results.append((lag, portfolio, backtest))
    return results


def _lag_analysis_from_backtests(
    lag_backtests: list[tuple[int, str, BacktestResult | None]],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for lag, portfolio, backtest in lag_backtests:
        row: dict[str, object] = {
            "lag": lag,
            "portfolio": portfolio,
            "gross_cumulative_return": math.nan,
            "net_cumulative_return": math.nan,
            "gross_sharpe": math.nan,
            "net_sharpe": math.nan,
        }
        if backtest is not None:
            row.update(
                {
                    "gross_cumulative_return": backtest.summary.gross_total_return,
                    "net_cumulative_return": backtest.summary.net_total_return,
                    "gross_sharpe": backtest.summary.gross_sharpe,
                    "net_sharpe": backtest.summary.net_sharpe,
                }
            )
        rows.append(row)
    return pl.DataFrame(rows).sort(["portfolio", "lag"])


def _lag_returns_from_backtests(
    lag_backtests: list[tuple[int, str, BacktestResult | None]],
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for lag, portfolio, backtest in lag_backtests:
        if backtest is None:
            continue
        frames.append(
            backtest.value.select(
                pl.lit(lag).alias("lag"),
                pl.lit(portfolio).alias("portfolio"),
                TIME,
                pl.col("gross_return_cumulative").alias("gross_cumulative_return"),
                pl.col("net_return_cumulative").alias("net_cumulative_return"),
                pl.lit(backtest.summary.gross_sharpe).alias("gross_sharpe"),
                pl.lit(backtest.summary.net_sharpe).alias("net_sharpe"),
            )
        )
    if not frames:
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
    return pl.concat(frames).sort(["portfolio", "lag", TIME])


def _spread_returns(quantile_returns: pl.DataFrame, quantiles: int) -> pl.DataFrame:
    top = "q1"
    bottom = f"q{quantiles}"
    return (
        quantile_returns.filter(pl.col("quantile").is_in([bottom, top]))
        .select(TIME, "quantile", "return")
        .pivot(index=TIME, on="quantile", values="return")
        .with_columns((pl.col(top) - pl.col(bottom)).alias("spread_return"))
        .select(TIME, "spread_return")
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
