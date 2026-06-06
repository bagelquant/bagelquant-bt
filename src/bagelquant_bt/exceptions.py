"""Package-specific exceptions for bagelquant-bt."""


class BagelQuantBacktestError(Exception):
    """Base exception for bagelquant-bt."""


class InputValidationError(BagelQuantBacktestError):
    """Raised when user-provided tabular inputs are invalid."""


class BacktestConfigError(BagelQuantBacktestError):
    """Raised when backtest configuration values are invalid."""
