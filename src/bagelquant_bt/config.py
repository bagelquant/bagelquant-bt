"""Configuration objects for backtesting and factor evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .exceptions import BacktestConfigError


@dataclass(frozen=True, slots=True)
class TransactionCostConfig:
    """Per-trade transaction cost settings."""

    rate: float = 0.00015
    min_fee: float = 5.0
    buy_slippage_rate: float = 0.0005
    sell_slippage_rate: float = 0.0005
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("rate", self.rate),
            ("min_fee", self.min_fee),
            ("buy_slippage_rate", self.buy_slippage_rate),
            ("sell_slippage_rate", self.sell_slippage_rate),
            ("stamp_tax_rate", self.stamp_tax_rate),
            ("transfer_fee_rate", self.transfer_fee_rate),
        ):
            if not math.isfinite(value) or value < 0:
                raise BacktestConfigError(
                    f"transaction cost {name} must be finite and nonnegative"
                )

    def slippage_for(self, side: Literal["buy", "sell"]) -> float:
        """Return the explicitly configured side-specific rate."""

        return self.buy_slippage_rate if side == "buy" else self.sell_slippage_rate


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
    insolvency_action: Literal["raise", "freeze_zero"] = "raise"

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
        if self.insolvency_action not in {"raise", "freeze_zero"}:
            raise BacktestConfigError(
                "insolvency_action must be 'raise' or 'freeze_zero'"
            )

    @property
    def resolved_ic_annualization(self) -> int:
        """Return the number of IC observations represented by one year."""

        return self.ic_annualization or self.annualization
