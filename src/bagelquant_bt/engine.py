"""Backtest orchestration."""

from __future__ import annotations

import polars as pl

from .config import BacktestConfig
from .costs import turnover
from .exceptions import BacktestConfigError, InputValidationError
from .inputs import ASSET_ID, TIME, validate_prices, validate_weights
from .performance import summarize_performance
from .results import BacktestResult, TransactionCostBreakdown
from .returns import (
    _expand_portfolio_weights,
    asset_close_to_close_returns,
    cumulative_returns,
    portfolio_returns,
)


def run_weight_backtest(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest a long-form portfolio weight frame."""

    resolved_config = _require_config(config)
    aligned_weights = validate_weights(weights)
    aligned_prices = validate_prices(prices)
    return backtest_weight_frame(
        aligned_weights,
        aligned_prices,
        config=resolved_config,
    )


def backtest_weight_frame(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
) -> BacktestResult:
    """Backtest an already materialized long-form weight frame."""

    aligned_weights = validate_weights(weights)
    aligned_prices = validate_prices(prices)
    forward_returns = asset_close_to_close_returns(aligned_prices)
    return _backtest_weight_frame_with_forward_returns(
        aligned_weights,
        aligned_prices,
        forward_returns,
        config=config,
    )


def _backtest_weight_frame_with_forward_returns(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
) -> BacktestResult:
    """Backtest a weight frame with a precomputed forward-return panel."""

    executable_weights = _expand_portfolio_weights(weights, prices, forward_returns)
    if executable_weights.is_empty():
        raise InputValidationError("at least two overlapping price times are required")

    gross_returns = portfolio_returns(executable_weights, forward_returns)
    turn = turnover(executable_weights)
    costs, returns, value = _simulate_cost_adjusted_returns(
        weights=executable_weights,
        gross_returns=gross_returns,
        config=config,
    )
    summary, performance = summarize_performance(
        returns=returns,
        turnover=turn,
        costs=costs,
        initial_capital=config.initial_capital,
        annualization=config.annualization,
    )
    return BacktestResult(
        weights=executable_weights,
        asset_returns=forward_returns,
        returns=returns,
        value=value,
        turnover=turn,
        transaction_costs=costs,
        summary=summary,
        performance=performance,
    )


def _simulate_cost_adjusted_returns(
    *,
    weights: pl.DataFrame,
    gross_returns: pl.DataFrame,
    config: BacktestConfig,
) -> tuple[TransactionCostBreakdown, pl.DataFrame, pl.DataFrame]:
    trade_summary = _trade_summary(weights)
    gross_by_time = {
        row[TIME]: float(row["gross_return"] or 0.0)
        for row in gross_returns.iter_rows(named=True)
    }

    current_value = float(config.initial_capital)
    cost_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []

    for row in trade_summary.iter_rows(named=True):
        time = row[TIME]
        weight_deltas = [float(delta) for delta in row["weight_deltas"]]
        traded_asset_count = len(weight_deltas)
        weight_delta = sum(weight_deltas)
        traded_notional = weight_delta * current_value
        raw_fee = traded_notional * config.transaction_cost.rate
        total_fee = sum(
            max(
                delta * current_value * config.transaction_cost.rate,
                config.transaction_cost.min_fee,
            )
            for delta in weight_deltas
        )

        cost_return = total_fee / current_value if current_value else 0.0
        gross_return = gross_by_time.get(time, 0.0)
        net_return = gross_return - cost_return
        gross_value = current_value * (1.0 + gross_return)
        current_value *= 1.0 + net_return
        cost_rows.append(
            {
                TIME: time,
                "traded_asset_count": traded_asset_count,
                "traded_notional": traded_notional,
                "raw_fee": raw_fee,
                "min_fee_adjustment": total_fee - raw_fee,
                "total_fee": total_fee,
                "cost_return": cost_return,
            }
        )
        return_rows.append(
            {TIME: time, "gross_return": gross_return, "net_return": net_return}
        )
        value_rows.append(
            {TIME: time, "gross_value": gross_value, "net_value": current_value}
        )

    returns = pl.DataFrame(return_rows).sort(TIME)
    value = (
        pl.DataFrame(value_rows)
        .sort(TIME)
        .join(
            cumulative_returns(returns, "gross_return"),
            on=TIME,
        )
        .join(
            cumulative_returns(returns, "net_return"),
            on=TIME,
        )
    )
    return TransactionCostBreakdown(pl.DataFrame(cost_rows).sort(TIME)), returns, value


def _trade_summary(weights: pl.DataFrame) -> pl.DataFrame:
    return (
        weights.sort([ASSET_ID, TIME])
        .with_columns(
            pl.col("weight")
            .fill_null(0.0)
            .shift(1)
            .over(ASSET_ID)
            .fill_null(0.0)
            .alias("previous_weight")
        )
        .with_columns(
            (pl.col("weight").fill_null(0.0) - pl.col("previous_weight"))
            .abs()
            .alias("weight_delta")
        )
        .group_by(TIME)
        .agg(
            pl.col("weight_delta")
            .filter(pl.col("weight_delta") > 0.0)
            .alias("weight_deltas"),
        )
        .sort(TIME)
    )


def _require_config(config: BacktestConfig | None) -> BacktestConfig:
    if config is None:
        raise BacktestConfigError(
            "config is required because initial_capital is needed for minimum fees"
        )
    return config


def _partition_key(key: object) -> object:
    if isinstance(key, tuple):
        return key[0]
    if isinstance(key, list):
        return key[0]
    return key
