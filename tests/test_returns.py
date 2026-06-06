from __future__ import annotations

import pandas as pd
import pytest

from bagelquant_bt.returns import (
    align_weights_to_forward_returns,
    asset_close_to_close_returns,
    portfolio_returns,
)


def test_close_to_close_returns() -> None:
    prices = pd.DataFrame(
        {"a": [100.0, 110.0, 121.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    returns = asset_close_to_close_returns(prices)

    assert pd.isna(returns["a"].iloc[0])
    assert returns["a"].iloc[1:].tolist() == pytest.approx([0.1, 0.1])


def test_no_lookahead_alignment_uses_weight_date_for_next_return() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame({"a": [100.0, 110.0, 99.0]}, index=dates)
    weights = pd.DataFrame({"a": [1.0, 0.0, 1.0]}, index=dates)

    aligned_weights, forward_returns = align_weights_to_forward_returns(weights, prices)
    result = portfolio_returns(aligned_weights, forward_returns)

    assert aligned_weights.index.tolist() == dates[:-1].tolist()
    assert result.round(6).tolist() == [0.1, 0.0]
