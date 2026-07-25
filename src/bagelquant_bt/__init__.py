"""Backtesting and factor evaluation for the BagelQuant ecosystem."""

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
    HolidayAdjustment,
    SignalAnchor,
    SignalFrequency,
    SignalPolicy,
    resolve_signal_policy,
    signal_policies,
)

__all__ = [
    "BacktestConfig",
    "BacktestConfigError",
    "BacktestResult",
    "BagelQuantBacktestError",
    "EqualWeightPolicy",
    "FactorEvaluationResult",
    "FloatMarketCapWeightPolicy",
    "HolidayAdjustment",
    "InputValidationError",
    "PerformanceSummary",
    "PortfolioBuild",
    "PreparedFactorMarketData",
    "ReportFigure",
    "SignalAnchor",
    "SignalFrequency",
    "SignalPolicy",
    "TargetVolatilityPolicy",
    "TransactionCostBreakdown",
    "TransactionCostConfig",
    "factor_evaluation_report_figures",
    "prepare_factor_market_data",
    "resolve_signal_policy",
    "run_factor_evaluation",
    "run_signal_evaluation",
    "run_weight_backtest",
    "signal_policies",
    "summary_report",
]
