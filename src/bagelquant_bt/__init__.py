"""Backtesting and factor evaluation for the BagelQuant ecosystem."""

from .benchmarks import benchmark_performance, build_universe_benchmark_returns
from .config import BacktestConfig, TransactionCostConfig
from .engine import run_weight_backtest
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
    run_factor_evaluation,
    run_signal_evaluation,
)
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
from .returns import prepare_price_data
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
    "SignalAnchor",
    "SignalFrequency",
    "SignalPolicy",
    "SignalSelection",
    "TargetVolatilityPolicy",
    "TransactionCostBreakdown",
    "TransactionCostConfig",
    "benchmark_performance",
    "build_universe_benchmark_returns",
    "compute_result_section",
    "execution_policies",
    "factor_evaluation_report_figures",
    "information_coefficients",
    "materialize_portfolio_path",
    "materialize_signal_diagnostics",
    "portfolio_path_from_backtest",
    "prepare_factor_market_data",
    "prepare_price_data",
    "resolve_execution_policy",
    "resolve_signal_policy",
    "resume_portfolio_path",
    "run_factor_evaluation",
    "run_signal_evaluation",
    "run_weight_backtest",
    "signal_policies",
    "summary_report",
]
