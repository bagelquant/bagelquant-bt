"""Transaction cost and turnover calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TransactionCostConfig
from .results import TransactionCostBreakdown


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute daily asset-level absolute weight turnover."""

    previous = weights.shift(1).fillna(0.0)
    current = weights.fillna(0.0)
    return current.sub(previous).abs().sum(axis=1)


def transaction_costs(
    weights: pd.DataFrame,
    *,
    portfolio_value_before_trade: pd.Series,
    config: TransactionCostConfig,
) -> TransactionCostBreakdown:
    """Apply percentage cost with a per-asset minimum fee floor."""

    previous = weights.shift(1).fillna(0.0)
    delta = weights.fillna(0.0).sub(previous).abs()
    notional = delta.mul(portfolio_value_before_trade, axis=0)
    traded = notional.gt(0.0)
    raw_fee_frame = notional * config.rate
    fee_frame = raw_fee_frame.where(~traded, np.maximum(raw_fee_frame, config.min_fee))
    fee_frame = fee_frame.where(traded, 0.0)

    raw_fee = raw_fee_frame.where(traded, 0.0).sum(axis=1)
    total_fee = fee_frame.sum(axis=1)
    traded_notional = notional.sum(axis=1)
    traded_asset_count = traded.sum(axis=1)
    min_fee_adjustment = total_fee.sub(raw_fee)
    cost_return = total_fee.div(portfolio_value_before_trade).fillna(0.0)

    return TransactionCostBreakdown(
        traded_asset_count=traded_asset_count,
        traded_notional=traded_notional,
        raw_fee=raw_fee,
        min_fee_adjustment=min_fee_adjustment,
        total_fee=total_fee,
        cost_return=cost_return,
    )
