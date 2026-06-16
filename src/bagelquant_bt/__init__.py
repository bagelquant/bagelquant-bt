"""Backtesting and factor evaluation for the BagelQuant ecosystem."""

from .config import BacktestConfig, TransactionCostConfig
from .engine import run_weight_backtest
from .exceptions import (
    BacktestConfigError,
    BagelQuantBacktestError,
    InputValidationError,
)
from .factor import run_factor_evaluation
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
    "TransactionCostBreakdown",
    "TransactionCostConfig",
    "run_factor_evaluation",
    "run_weight_backtest",
    "summary_report",
]
