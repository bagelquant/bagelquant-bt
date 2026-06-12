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
    align_weights_to_forward_returns,
    cumulative_returns,
    portfolio_returns,
)


def run_backtest(
    signal: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    kind: str,
    config: BacktestConfig | None = None,
) -> BacktestResult | object:
    """Dispatch to weight backtest or factor evaluation based on explicit kind."""

    if kind == "weights":
        return run_weight_backtest(signal, prices, config=config)
    if kind == "factor":
        from .factor import run_factor_evaluation

        return run_factor_evaluation(signal, prices, config=config)
    raise InputValidationError("kind must be 'weights' or 'factor'")


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
    executable_weights, forward_returns = align_weights_to_forward_returns(
        aligned_weights,
        aligned_prices,
    )
    if executable_weights.is_empty():
        raise InputValidationError("at least two overlapping price times are required")

    gross_returns = portfolio_returns(executable_weights, forward_returns)
    turn = turnover(executable_weights)
    costs, returns, value = _simulate_cost_adjusted_returns(
        weights=executable_weights,
        gross_returns=gross_returns,
        config=config,
    )
    summary = summarize_performance(
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
    )


def _simulate_cost_adjusted_returns(
    *,
    weights: pl.DataFrame,
    gross_returns: pl.DataFrame,
    config: BacktestConfig,
) -> tuple[TransactionCostBreakdown, pl.DataFrame, pl.DataFrame]:
    weights_by_time = {
        _partition_key(time): {
            str(row[ASSET_ID]): 0.0 if row["weight"] is None else float(row["weight"])
            for row in group.iter_rows(named=True)
        }
        for time, group in weights.partition_by(TIME, as_dict=True).items()
    }
    gross_by_time = {
        row[TIME]: float(row["gross_return"] or 0.0)
        for row in gross_returns.iter_rows(named=True)
    }

    current_value = float(config.initial_capital)
    previous_weights: dict[str, float] = {}
    cost_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []

    for time in sorted(weights_by_time):
        current_weights = weights_by_time[time]
        assets = set(previous_weights) | set(current_weights)
        traded_asset_count = 0
        traded_notional = 0.0
        raw_fee = 0.0
        total_fee = 0.0
        for asset in assets:
            delta = abs(
                current_weights.get(asset, 0.0) - previous_weights.get(asset, 0.0)
            )
            notional = delta * current_value
            if notional <= 0.0:
                continue
            traded_asset_count += 1
            traded_notional += notional
            asset_raw_fee = notional * config.transaction_cost.rate
            raw_fee += asset_raw_fee
            total_fee += max(asset_raw_fee, config.transaction_cost.min_fee)

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
        previous_weights = current_weights

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
