from __future__ import annotations

import math

import polars as pl
import pytest

from bagelquant_bt import cross_sectional_factor_returns, one_sample_t_test


def test_cross_sectional_factor_returns_match_hand_checked_ols_slopes() -> None:
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 3 + ["2024-02-01"] * 3,
            "asset_id": ["a", "b", "c"] * 2,
            "factor": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    returns = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 3 + ["2024-02-01"] * 3,
            "asset_id": ["a", "b", "c"] * 2,
            "forward_return": [0.1, 0.2, 0.3, -0.2, 0.0, 0.2],
        }
    )

    result = cross_sectional_factor_returns(factor, returns)

    assert result.get_column("lambda_return").to_list() == pytest.approx([0.1, 0.2])
    assert result.get_column("sample_size").to_list() == [3, 3]


def test_one_sample_t_test_reports_exact_student_t_and_edge_states() -> None:
    result = one_sample_t_test([1.0, None, 2.0, float("nan"), 3.0])
    assert result.mean == pytest.approx(2.0)
    assert result.t_value == pytest.approx(math.sqrt(12.0))
    assert result.p_value == pytest.approx(0.0741799002)
    assert result.sample_size == 3
    assert result.reason is None

    insufficient = one_sample_t_test([1.0])
    assert insufficient.t_value is None
    assert insufficient.reason == "at least two samples required"

    constant = one_sample_t_test([1.0, 1.0])
    assert constant.p_value is None
    assert constant.reason == "sample variance is zero"
