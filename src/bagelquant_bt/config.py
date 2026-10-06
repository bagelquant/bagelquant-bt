"""Neutral money-cost parameters shared by execution primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .exceptions import BacktestConfigError


@dataclass(frozen=True, slots=True)
class TransactionCostConfig:
    """Per-trade transaction cost settings."""

    rate: float = 0.0005
    min_fee: float = 5.0
    buy_slippage_rate: float = 0.0
    sell_slippage_rate: float = 0.0
    stamp_tax_rate: float = 0.0
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
