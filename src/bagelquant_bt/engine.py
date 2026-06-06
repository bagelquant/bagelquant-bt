"""Backtest orchestration."""

from __future__ import annotations

import pandas as pd

from .config import BacktestConfig
from .costs import turnover
from .exceptions import BacktestConfigError, InputValidationError
from .inputs import align_signal_and_prices
from .performance import summarize_performance
from .results import BacktestResult, TransactionCostBreakdown
from .returns import (
    align_weights_to_forward_returns,
    cumulative_returns,
    portfolio_returns,
    value_path,
)


def run_backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
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
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest a portfolio weight DataFrame."""

    resolved_config = _require_config(config)
    aligned_weights, aligned_prices = align_signal_and_prices(weights, prices)
    return backtest_weight_frame(
        aligned_weights,
        aligned_prices,
        config=resolved_config,
    )


def backtest_weight_frame(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> BacktestResult:
    """Backtest an already materialized weight frame."""

    aligned_weights, aligned_prices = align_signal_and_prices(weights, prices)
    executable_weights, forward_returns = align_weights_to_forward_returns(
        aligned_weights,
        aligned_prices,
    )
    if executable_weights.empty:
        raise InputValidationError("at least two overlapping price dates are required")

    gross_returns = portfolio_returns(executable_weights, forward_returns)
    turn = turnover(executable_weights)
    costs, net_returns, gross_value, net_value = _simulate_cost_adjusted_returns(
        weights=executable_weights,
        gross_returns=gross_returns,
        config=config,
    )
    summary = summarize_performance(
        gross_returns=gross_returns,
        net_returns=net_returns,
        turnover=turn,
        costs=costs,
        initial_capital=config.initial_capital,
        annualization=config.annualization,
    )
    return BacktestResult(
        weights=executable_weights,
        asset_returns=forward_returns,
        gross_returns=gross_returns,
        net_returns=net_returns,
        gross_cumulative_returns=cumulative_returns(gross_returns),
        net_cumulative_returns=cumulative_returns(net_returns),
        gross_value=gross_value,
        net_value=net_value,
        turnover=turn,
        transaction_costs=costs,
        summary=summary,
    )


def _simulate_cost_adjusted_returns(
    *,
    weights: pd.DataFrame,
    gross_returns: pd.Series,
    config: BacktestConfig,
) -> tuple[TransactionCostBreakdown, pd.Series, pd.Series, pd.Series]:
    dates = weights.index
    value_before_trade = pd.Series(index=dates, dtype=float)
    net_returns = pd.Series(index=dates, dtype=float)
    net_value = pd.Series(index=dates, dtype=float)

    current_value = float(config.initial_capital)
    previous_weights = pd.Series(0.0, index=weights.columns)
    traded_asset_count = pd.Series(index=dates, dtype=int)
    traded_notional = pd.Series(index=dates, dtype=float)
    raw_fee = pd.Series(index=dates, dtype=float)
    min_fee_adjustment = pd.Series(index=dates, dtype=float)
    total_fee = pd.Series(index=dates, dtype=float)
    cost_return = pd.Series(index=dates, dtype=float)

    for date in dates:
        value_before_trade.loc[date] = current_value
        current_weights = weights.loc[date].fillna(0.0)
        delta = current_weights.sub(previous_weights).abs()
        notional = delta * current_value
        traded = notional.gt(0.0)
        raw_fee_row = notional * config.transaction_cost.rate
        fee_row = raw_fee_row.where(
            ~traded,
            raw_fee_row.clip(lower=config.transaction_cost.min_fee),
        ).where(traded, 0.0)

        traded_asset_count.loc[date] = int(traded.sum())
        traded_notional.loc[date] = float(notional.sum())
        raw_fee.loc[date] = float(raw_fee_row.where(traded, 0.0).sum())
        total_fee.loc[date] = float(fee_row.sum())
        min_fee_adjustment.loc[date] = total_fee.loc[date] - raw_fee.loc[date]
        cost_return.loc[date] = total_fee.loc[date] / current_value

        net_return = float(gross_returns.loc[date]) - cost_return.loc[date]
        net_returns.loc[date] = net_return
        current_value *= 1.0 + net_return
        net_value.loc[date] = current_value
        previous_weights = current_weights

    costs = TransactionCostBreakdown(
        traded_asset_count=traded_asset_count,
        traded_notional=traded_notional,
        raw_fee=raw_fee,
        min_fee_adjustment=min_fee_adjustment,
        total_fee=total_fee,
        cost_return=cost_return,
    )
    gross_value = value_path(gross_returns, initial_capital=config.initial_capital)
    return costs, net_returns, gross_value, net_value


def _require_config(config: BacktestConfig | None) -> BacktestConfig:
    if config is None:
        raise BacktestConfigError(
            "config is required because initial_capital is needed for minimum fees"
        )
    return config
