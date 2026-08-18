from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from bagelquant_bt import (
    cross_sectional_factor_returns,
    one_sample_t_test,
    quantile_rank_information_coefficients,
)


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


def test_quantile_rank_ic_supports_complete_current_and_historical_groups() -> None:
    current = pl.DataFrame(
        {
            "time": [date(2024, 1, 31)] * 10 + [date(2024, 2, 29)] * 10,
            "quantile": [f"q{number}" for number in range(1, 11)] * 2,
            "return": list(range(10, 0, -1)) + list(range(1, 11)),
        }
    )
    historical = pl.DataFrame(
        {
            "time": [date(2023, 11, 30)] * 4 + [date(2023, 12, 29)] * 5,
            "quantile": ["q1", "q2", "q3", "q4"]
            + [f"q{number}" for number in range(1, 6)],
            "return": [5.0, 4.0, 3.0, 2.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    result = quantile_rank_information_coefficients(current)
    legacy = quantile_rank_information_coefficients(historical)

    assert result.get_column("quantile_rank_ic").to_list() == pytest.approx(
        [1.0, -1.0]
    )
    assert result.get_column("quantiles").to_list() == [10, 10]
    assert legacy.item(0, "quantile_rank_ic") is None
    assert legacy.item(1, "quantile_rank_ic") == pytest.approx(1.0)
    assert legacy.get_column("quantiles").to_list() == [5, 5]


def test_quantile_rank_ic_compounds_daily_returns_by_execution_period() -> None:
    days = [
        date(2024, 1, 31),
        date(2024, 2, 1),
        date(2024, 2, 2),
        date(2024, 2, 3),
    ]
    rows = []
    for day in days[:2]:
        rows.extend(
            {
                "time": day,
                "quantile": f"q{number}",
                "return": (6 - number) / 100,
            }
            for number in range(1, 6)
        )
    for day in days[2:]:
        rows.extend(
            {
                "time": day,
                "quantile": f"q{number}",
                "return": number / 100,
            }
            for number in range(1, 6)
        )
    periods = pl.DataFrame(
        {
            "time": [days[0], days[2]],
            "next_time": [days[2], date(2024, 2, 4)],
        }
    )

    result = quantile_rank_information_coefficients(
        pl.DataFrame(rows),
        periods=periods,
    )

    assert result.get_column("time").to_list() == [days[0], days[2]]
    assert result.get_column("quantile_rank_ic").to_list() == pytest.approx(
        [1.0, -1.0]
    )


def test_periodized_quantile_rank_ic_nulls_incomplete_and_constant_groups() -> None:
    start = date(2024, 1, 31)
    next_time = date(2024, 2, 2)
    periods = pl.DataFrame({"time": [start], "next_time": [next_time]})
    incomplete = pl.DataFrame(
        {
            "time": [start] * 5,
            "quantile": [f"q{number}" for number in range(1, 6)],
            "return": [0.05, 0.04, 0.03, 0.02, None],
        }
    )
    constant = incomplete.with_columns(pl.lit(0.01).alias("return"))

    assert quantile_rank_information_coefficients(
        incomplete, periods=periods
    ).item(0, "quantile_rank_ic") is None
    assert quantile_rank_information_coefficients(
        constant, periods=periods
    ).item(0, "quantile_rank_ic") is None
