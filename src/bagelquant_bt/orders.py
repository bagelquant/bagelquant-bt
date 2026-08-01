"""Reserved target-weight-to-order sizing boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bagelquant_core import Panel


@dataclass(frozen=True, slots=True)
class OrderPlan:
    """Future order quantities plus explicit skipped-order diagnostics."""

    quantities: Panel
    skipped: object


class OrderSizingPolicy(Protocol):
    """Future contract for converting weights and holdings into quantities."""

    def build(
        self,
        target_weights: Panel,
        holdings: Panel,
        prices: Panel,
        *,
        capital: float,
    ) -> OrderPlan: ...


__all__ = ["OrderPlan", "OrderSizingPolicy"]
