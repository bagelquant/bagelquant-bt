"""Result containers returned by bagelquant-bt."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class TransactionCostBreakdown:
    """Daily transaction cost details."""

    traded_asset_count: pd.Series
    traded_notional: pd.Series
    raw_fee: pd.Series
    min_fee_adjustment: pd.Series
    total_fee: pd.Series
    cost_return: pd.Series


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """High-level performance metrics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    average_turnover: float
    total_transaction_cost: float
    final_gross_value: float
    final_net_value: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Portfolio backtest result with gross and net return paths."""

    weights: pd.DataFrame
    asset_returns: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    gross_cumulative_returns: pd.Series
    net_cumulative_returns: pd.Series
    gross_value: pd.Series
    net_value: pd.Series
    turnover: pd.Series
    transaction_costs: TransactionCostBreakdown
    summary: PerformanceSummary


@dataclass(frozen=True, slots=True)
class FactorEvaluationResult:
    """Factor diagnostics and derived TOP N backtest."""

    factor: pd.DataFrame
    forward_returns: pd.DataFrame
    ic: pd.Series
    ic_mean: float
    ic_std: float
    icir: float
    quantile_returns: pd.DataFrame
    quantile_cumulative_returns: pd.DataFrame
    top_minus_bottom: pd.Series
    top_n_weights: pd.DataFrame
    top_n_backtest: BacktestResult
