"""Backtesting and factor evaluation for the BagelQuant ecosystem."""

from .benchmarks import build_universe_benchmark_returns
from .config import BacktestConfig, TransactionCostConfig
from .engine import run_weight_backtest
from .exceptions import (
    BacktestConfigError,
    BagelQuantBacktestError,
    InputValidationError,
)
from .factor import (
    PreparedFactorMarketData,
    prepare_factor_market_data,
    run_factor_evaluation,
    run_signal_evaluation,
)
from .portfolio import (
    EqualWeightPolicy,
    FloatMarketCapWeightPolicy,
    PortfolioBuild,
    TargetVolatilityPolicy,
)
from .reporting import ReportFigure, factor_evaluation_report_figures, summary_report
from .results import (
    BacktestResult,
    FactorEvaluationResult,
    PerformanceSummary,
    TransactionCostBreakdown,
)
from .signal import (
    ExecutionPolicy,
    HolidayAdjustment,
    MissingSnapshotAction,
    SignalAnchor,
    SignalFrequency,
    SignalPolicy,
    SignalSelection,
    execution_policies,
    resolve_execution_policy,
    resolve_signal_policy,
    signal_policies,
)

__all__ = [
    "BacktestConfig",
    "BacktestConfigError",
    "BacktestResult",
    "BagelQuantBacktestError",
    "EqualWeightPolicy",
    "ExecutionPolicy",
    "FactorEvaluationResult",
    "FloatMarketCapWeightPolicy",
    "HolidayAdjustment",
    "InputValidationError",
    "MissingSnapshotAction",
    "PerformanceSummary",
    "PortfolioBuild",
    "PreparedFactorMarketData",
    "ReportFigure",
    "SignalAnchor",
    "SignalFrequency",
    "SignalPolicy",
    "SignalSelection",
    "TargetVolatilityPolicy",
    "TransactionCostBreakdown",
    "TransactionCostConfig",
    "build_universe_benchmark_returns",
    "execution_policies",
    "factor_evaluation_report_figures",
    "prepare_factor_market_data",
    "resolve_execution_policy",
    "resolve_signal_policy",
    "run_factor_evaluation",
    "run_signal_evaluation",
    "run_weight_backtest",
    "signal_policies",
    "summary_report",
]
