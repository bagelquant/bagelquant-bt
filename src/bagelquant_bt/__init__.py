"""Backtesting and factor evaluation for the BagelQuant ecosystem."""

from .config import BacktestConfig, TransactionCostConfig
from .engine import run_weight_backtest
from .exceptions import (
    BacktestConfigError,
    BagelQuantBacktestError,
    InputValidationError,
)
from .factor import PreparedFactorMarketData, prepare_factor_market_data, run_factor_evaluation
from .reporting import summary_report
from .results import (
    BacktestResult,
    FactorEvaluationResult,
    PerformanceSummary,
    TransactionCostBreakdown,
)

__all__ = [
    "BacktestConfig",
    "BacktestConfigError",
    "BacktestResult",
    "BagelQuantBacktestError",
    "FactorEvaluationResult",
    "InputValidationError",
    "PerformanceSummary",
    "PreparedFactorMarketData",
    "TransactionCostBreakdown",
    "TransactionCostConfig",
    "run_factor_evaluation",
    "prepare_factor_market_data",
    "run_weight_backtest",
    "summary_report",
]
