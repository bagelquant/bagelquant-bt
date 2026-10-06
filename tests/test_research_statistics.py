from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from bagelquant_bt import (
    account_fill_turnover,
    capacity_participation,
    common_alpha_ic,
    common_sample_comparison,
    deflated_sharpe,
    library_endpoint_tests,
)


def test_alpha_comparison_uses_identical_asset_sample_and_caller_labels():
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(9)]
    values = pl.DataFrame(
        {
            "time": [days[0]] * 8,
            "asset_id": [str(index) for index in range(8)],
            "value": list(map(float, range(8))),
        }
    )
    variant = values.head(7)
    labels = pl.DataFrame(
        [
            {
                "time": day,
                "asset_id": str(asset),
                "forward_return": 0.01 * (asset + 1),
                "interval_start": days[index + 1],
                "interval_end": days[index + 2],
                "available_date": days[index + 2],
            }
            for index, day in enumerate(days[:-2])
            for asset in range(8)
        ]
    )
    result = common_alpha_ic(
        values,
        variant,
        labels,
        calendar=pl.DataFrame({"time": days}),
        horizon=2,
        batch_sessions=1,
    )
    assert result["sample_count"].to_list() == [7]
    assert result["spearman_ic_increment"].to_list() == [0.0]
    assert result["conditional_ic"].to_list() == [None]
    assert result["reason"].to_list() == ["no_independent_signal"]


def test_account_turnover_uses_previous_equity_and_actual_fills():
    days = [date(2024, 1, 1), date(2024, 1, 2)]
    fills = pl.DataFrame({"time": days, "notional": [100.0, -50.0]})
    equity = pl.DataFrame({"time": days, "equity": [2000.0, 3000.0]})
    result = account_fill_turnover(fills, equity, initial_capital=1000.0)
    assert result["gross_traded_fraction"].to_list() == [0.1, 0.025]


def test_bh_includes_abandoned_trials_and_no_statistic_is_not_a_pvalue():
    rows = library_endpoint_tests(
        [
            {
                "id": 1,
                "statistic": {"p_value": 0.01, "mean": 0.02},
                "direction": "positive",
                "minimum_effect": 0.01,
            },
            {
                "id": 2,
                "state": "abandoned",
                "statistic": {"p_value": 0.04, "mean": -0.01},
                "direction": "positive",
            },
            {"id": 3, "state": "failed", "statistic": {}},
        ]
    )
    assert rows[0]["q_value"] == pytest.approx(0.02)
    assert rows[1]["q_value"] == pytest.approx(0.04)
    assert rows[0]["passes"] and not rows[1]["passes"]
    assert rows[2]["q_value"] is None and rows[2]["family_size"] == 2


def test_comparison_uses_common_finite_dates():
    baseline = pl.DataFrame({"time": [1, 2, 3], "return": [0.1, None, 0.3]})
    variant = pl.DataFrame({"time": [2, 3, 4], "return": [0.2, 0.4, 0.5]})
    frame, summary = common_sample_comparison(baseline, variant, columns=["return"])
    assert frame["time"].to_list() == [3]
    assert frame["return_increment"][0] == pytest.approx(0.1)
    assert summary["increments"]["return"]["p_value"] is None


def test_capacity_excludes_execution_day_and_requires_all_twenty_sessions():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(22)]
    turnover = pl.DataFrame(
        {"time": days, "asset_id": ["A"] * 22, "amount": [100.0] * 21 + [1000000.0]}
    )
    fills = pl.DataFrame(
        {
            "time": [days[19], days[20], days[21]],
            "asset_id": ["A"] * 3,
            "notional": [10.0] * 3,
        }
    )
    result = capacity_participation(
        fills, turnover, calendar=pl.DataFrame({"time": days})
    )
    assert result["participation"].to_list() == [None, 0.1, 0.1]
    missing = capacity_participation(
        fills,
        turnover.filter(pl.col("time") != days[10]),
        calendar=pl.DataFrame({"time": days}),
    )
    assert missing["participation"].null_count() == 3


@pytest.mark.parametrize("invalid", [float("inf"), float("nan"), None, -1.0])
def test_capacity_invalid_observations_do_not_fabricate_a_twenty_session_mean(invalid):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(21)]
    amounts = [100.0] * 21
    amounts[10] = invalid
    result = capacity_participation(
        pl.DataFrame({"time": [days[-1]], "asset_id": ["A"], "notional": [10.0]}),
        pl.DataFrame({"time": days, "asset_id": ["A"] * 21, "amount": amounts}),
        calendar=pl.DataFrame({"time": days}),
    )
    assert result["adv20"].to_list() == [None]
    assert result["participation"].to_list() == [None]
    assert result["unavailable_reason"].to_list() == ["incomplete_prior_20_sessions"]


def test_capacity_accepts_observed_zero_and_rejects_zero_mean_liquidity():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(21)]
    fills = pl.DataFrame({"time": [days[-1]], "asset_id": ["A"], "notional": [10.0]})
    turnover = pl.DataFrame(
        {
            "time": days,
            "asset_id": ["A"] * 21,
            "amount": [0.0] + [100.0] * 19 + [1000000.0],
        }
    )
    result = capacity_participation(
        fills, turnover, calendar=pl.DataFrame({"time": days})
    )
    assert result["adv20"].to_list() == [95.0]
    assert result["participation"][0] == pytest.approx(10 / 95)
    assert result["unavailable_reason"].to_list() == [None]
    zero = capacity_participation(
        fills,
        turnover.with_columns(pl.lit(0.0).alias("amount")),
        calendar=pl.DataFrame({"time": days}),
    )
    assert zero["adv20"].to_list() == [0.0]
    assert zero["participation"].to_list() == [None]
    assert zero["unavailable_reason"].to_list() == ["zero_prior_20_session_turnover"]


def test_deflated_sharpe_rejects_unsupported_trial_correlation_estimates():
    # A and B are identical, C is orthogonal: average rho=1/3 would imply
    # 7/3 independent trials, even though the correlation matrix is singular.
    rows = [
        {"time": day, "trial": name, "return": value}
        for day in range(300)
        for name, value in (
            ("A", float((-1, 1)[day % 2])),
            ("B", float((-1, 1)[day % 2])),
            ("C", float((-1, -1, 1, 1)[day % 4])),
        )
    ]
    returns = pl.DataFrame(rows)
    result = deflated_sharpe(returns, target="A", trials=list("ABC"))
    assert result["status"] == "unavailable"
    assert result["mean_trial_correlation"] == pytest.approx(1 / 3)
    assert result["reason"] == "ill_conditioned_trial_correlation"
    result = deflated_sharpe(
        returns.filter(pl.col("time") < 3),
        target="A",
        trials=list("ABC"),
        minimum_sessions=3,
    )
    assert result["reason"] == "insufficient_common_sessions_for_trial_estimate"


def test_deflated_sharpe_retains_scope_and_reports_insufficient_data():
    rng = np.random.default_rng(42)
    common = rng.normal(size=300)
    rows = [
        {
            "time": day,
            "trial": name,
            "return": float(
                0.003 * (i + 1) + 0.003 * common[day] + 0.01 * rng.normal()
            ),
        }
        for i, name in enumerate("ABC")
        for day in range(300)
    ]
    returns = pl.DataFrame(rows)
    result = deflated_sharpe(returns, target="C", trials=list("ABC"))
    assert result["status"] == "ready"
    assert 0 <= result["deflated_sharpe_probability"] <= 1
    assert result["effective_trials"] >= 2
    assert (
        deflated_sharpe(returns.head(10), target="C", trials=list("ABC"))["status"]
        == "unavailable"
    )
