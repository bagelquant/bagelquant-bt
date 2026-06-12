"""Result containers returned by bagelquant-bt."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class TransactionCostBreakdown:
    """Daily transaction cost details."""

    data: pl.DataFrame


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

    weights: pl.DataFrame
    asset_returns: pl.DataFrame
    returns: pl.DataFrame
    value: pl.DataFrame
    turnover: pl.DataFrame
    transaction_costs: TransactionCostBreakdown
    summary: PerformanceSummary


@dataclass(frozen=True, slots=True)
class FactorEvaluationResult:
    """Factor diagnostics and derived TOP N backtest."""

    factor: pl.DataFrame
    forward_returns: pl.DataFrame
    ic: pl.DataFrame
    ic_mean: float
    ic_std: float
    icir: float
    quantile_returns: pl.DataFrame
    top_minus_bottom: pl.DataFrame
    top_n_weights: pl.DataFrame
    top_n_backtest: BacktestResult
