"""Typed signal composition, evaluation, and backtesting for BagelQuant."""

from .benchmarks import benchmark_performance, build_universe_benchmark_returns
from .config import BacktestConfig, TransactionCostConfig
from .exceptions import (
    BacktestConfigError,
    BagelQuantBacktestError,
    InputValidationError,
)
from .factor import (
    PreparedFactorMarketData,
    information_coefficients,
    materialize_signal_diagnostics,
    prepare_factor_market_data,
    run_signal_evaluation,
)
from .orders import OrderPlan, OrderSizingPolicy
from .path import (
    RESULT_SECTIONS,
    PortfolioPathChunk,
    PortfolioPathIdentity,
    PortfolioStateCheckpoint,
    ResultSection,
    ResultSectionSpec,
    ResultWindow,
    compute_result_section,
    materialize_portfolio_path,
    portfolio_path_from_backtest,
    resume_portfolio_path,
)
from .pipeline import compose_signal, run_signal_backtest
from .portfolio import (
    EqualWeightPolicy,
    FloatMarketCapWeightPolicy,
    PortfolioBuild,
    TargetVolatilityPolicy,
)
from .reporting import ReportFigure, signal_evaluation_report_figures, summary_report
from .results import (
    BacktestResult,
    FactorEvaluationResult,
    PerformanceSummary,
    SignalEvaluationResult,
    TransactionCostBreakdown,
)
from .returns import prepare_price_data
from .signal import (
    ExecutionPolicy,
    HolidayAdjustment,
    MissingSnapshotAction,
    ScheduledSignal,
    SignalAnchor,
    SignalDatePolicy,
    SignalFrequency,
    execution_policies,
    resolve_execution_policy,
    resolve_signal_date_policy,
    signal_date_policies,
)

__all__ = [
    "RESULT_SECTIONS",
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
    "OrderPlan",
    "OrderSizingPolicy",
    "PerformanceSummary",
    "PortfolioBuild",
    "PortfolioPathChunk",
    "PortfolioPathIdentity",
    "PortfolioStateCheckpoint",
    "PreparedFactorMarketData",
    "ReportFigure",
    "ResultSection",
    "ResultSectionSpec",
    "ResultWindow",
    "ScheduledSignal",
    "SignalAnchor",
    "SignalDatePolicy",
    "SignalEvaluationResult",
    "SignalFrequency",
    "TargetVolatilityPolicy",
    "TransactionCostBreakdown",
    "TransactionCostConfig",
    "benchmark_performance",
    "build_universe_benchmark_returns",
    "compose_signal",
    "compute_result_section",
    "execution_policies",
    "information_coefficients",
    "materialize_portfolio_path",
    "materialize_signal_diagnostics",
    "portfolio_path_from_backtest",
    "prepare_factor_market_data",
    "prepare_price_data",
    "resolve_execution_policy",
    "resolve_signal_date_policy",
    "resume_portfolio_path",
    "run_signal_backtest",
    "run_signal_evaluation",
    "signal_date_policies",
    "signal_evaluation_report_figures",
    "summary_report",
]
