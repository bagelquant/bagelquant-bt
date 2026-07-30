"""Configuration objects for backtesting and factor evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .exceptions import BacktestConfigError


@dataclass(frozen=True, slots=True)
class TransactionCostConfig:
    """Per-trade transaction cost settings."""

    rate: float = 0.00015
    min_fee: float = 5.0

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise BacktestConfigError("transaction cost rate must be nonnegative")
        if self.min_fee < 0:
            raise BacktestConfigError("transaction cost min_fee must be nonnegative")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Shared backtest and factor evaluation configuration."""

    initial_capital: float
    transaction_cost: TransactionCostConfig = field(
        default_factory=TransactionCostConfig
    )
    annualization: int = 252
    ic_annualization: int | None = None
    ic_method: str = "spearman"
    quantiles: int = 5
    top_n: int = 50
    retry_blocked_orders: bool = True

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise BacktestConfigError("initial_capital must be positive")
        if self.annualization <= 0:
            raise BacktestConfigError("annualization must be positive")
        if self.ic_annualization is not None and self.ic_annualization <= 0:
            raise BacktestConfigError("ic_annualization must be positive")
        if self.ic_method not in {"spearman", "pearson"}:
            raise BacktestConfigError("ic_method must be 'spearman' or 'pearson'")
        if self.quantiles < 2:
            raise BacktestConfigError("quantiles must be at least 2")
        if self.top_n <= 0:
            raise BacktestConfigError("top_n must be positive")

    @property
    def resolved_ic_annualization(self) -> int:
        """Return the number of IC observations represented by one year."""

        return self.ic_annualization or self.annualization
