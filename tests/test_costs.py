from __future__ import annotations

import pandas as pd
import pytest

from bagelquant_bt.config import TransactionCostConfig
from bagelquant_bt.costs import transaction_costs, turnover


def test_turnover_uses_absolute_weight_changes() -> None:
    weights = pd.DataFrame(
        {"a": [0.5, 0.2], "b": [0.5, 0.8]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    assert turnover(weights).round(6).tolist() == [1.0, 0.6]


def test_transaction_cost_rate_and_minimum_fee_floor() -> None:
    weights = pd.DataFrame(
        {"a": [0.1], "b": [0.5]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    values = pd.Series([10_000.0], index=weights.index)

    costs = transaction_costs(
        weights,
        portfolio_value_before_trade=values,
        config=TransactionCostConfig(rate=0.00015, min_fee=5.0),
    )

    assert costs.traded_asset_count.iloc[0] == 2
    assert costs.raw_fee.iloc[0] == pytest.approx(0.9)
    assert costs.total_fee.iloc[0] == 10.0
    assert costs.min_fee_adjustment.iloc[0] == pytest.approx(9.1)
