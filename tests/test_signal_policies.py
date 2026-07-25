from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bagelquant_bt import (
    BacktestConfig,
    EqualWeightPolicy,
    FloatMarketCapWeightPolicy,
    TargetVolatilityPolicy,
    resolve_signal_policy,
    run_signal_evaluation,
)
from bagelquant_bt.exceptions import InputValidationError


def _calendar() -> pl.DataFrame:
    return pl.DataFrame(
        {"time": pl.date_range(date(2024, 1, 1), date(2024, 3, 31), "1d", eager=True)}
    ).filter(pl.col("time").dt.weekday() <= 5)


def test_month_end_signal_uses_fifteenth_open_session() -> None:
    predictions = pl.DataFrame(
        {
            "time": [
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 2, 29),
                date(2024, 2, 29),
            ],
            "asset_id": ["a", "b", "a", "b"],
            "prediction": [2.0, 1.0, 1.0, 2.0],
        }
    )

    signals = resolve_signal_policy("month_end").transform(predictions, _calendar())
    assert signals.get_column("time").unique().to_list() == [
        date(2024, 2, 21),
        date(2024, 3, 21),
    ]


def test_month_end_signal_skips_holidays_when_counting_execution_sessions() -> None:
    predictions = pl.DataFrame(
        {
            "time": [date(2024, 1, 31), date(2024, 1, 31)],
            "asset_id": ["a", "b"],
            "prediction": [2.0, 1.0],
        }
    )
    calendar = _calendar().filter(pl.col("time") != date(2024, 2, 12))

    signals = resolve_signal_policy("month_end").transform(predictions, calendar)

    assert signals.get_column("time").unique().to_list() == [date(2024, 2, 22)]


def test_month_end_schedule_omits_observations_without_future_sessions() -> None:
    calendar = _calendar().filter(pl.col("time") <= date(2024, 2, 9))

    schedule = resolve_signal_policy("month_end").schedule(calendar)

    assert schedule.is_empty()


def test_signal_policy_preserves_all_monthly_schedule_variants() -> None:
    calendar = pl.DataFrame(
        {
            "time": pl.date_range(date(2024, 1, 1), date(2024, 2, 5), "1d", eager=True)
        }
    ).with_columns((pl.col("time").dt.weekday() <= 5).cast(pl.Int8).alias("is_open"))

    mid = resolve_signal_policy("monthly_mid").schedule(calendar)
    monday = resolve_signal_policy("monthly_first_monday").schedule(calendar)
    friday = resolve_signal_policy("monthly_last_friday").schedule(calendar)

    assert mid.row(0, named=True)["observation_time"] == date(2024, 1, 15)
    assert monday.row(0, named=True)["observation_time"] == date(2024, 1, 1)
    assert friday.row(0, named=True)["observation_time"] == date(2024, 1, 26)

def test_signal_evaluation_uses_next_signal_horizon_and_daily_holding() -> None:
    signals = pl.DataFrame(
        {
            "time": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 3),
                date(2024, 1, 3),
            ],
            "asset_id": ["a", "b", "a", "b"],
            "signal": [2.0, 1.0, 1.0, 2.0],
        }
    )
    prices = pl.DataFrame(
        {
            "time": [date(2024, 1, day) for day in range(1, 5)] * 2,
            "asset_id": ["a"] * 4 + ["b"] * 4,
            "price": [10.0, 11.0, 12.0, 12.0, 10.0, 9.0, 8.0, 8.0],
        }
    )

    result = run_signal_evaluation(
        signals,
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )

    assert result.forward_returns.to_dicts() == [
        {
            "time": date(2024, 1, 1),
            "asset_id": "a",
            "forward_return": pytest.approx(0.2),
        },
        {
            "time": date(2024, 1, 1),
            "asset_id": "b",
            "forward_return": pytest.approx(-0.2),
        },
    ]
    assert result.ic.height == 1
    assert result.top_n_backtest.returns["time"].to_list() == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    assert result.top_n_backtest.transaction_costs.data["total_fee"].to_list()[1] == 0.0


def test_portfolio_policies_use_explicit_market_inputs() -> None:
    signals = pl.DataFrame(
        {"time": [date(2024, 1, 1)] * 2, "asset_id": ["a", "b"], "signal": [2.0, 1.0]}
    )
    caps = pl.DataFrame(
        {
            "time": [date(2024, 1, 1)] * 2,
            "asset_id": ["a", "b"],
            "float_market_cap": [1.0, 3.0],
        }
    )

    weights = FloatMarketCapWeightPolicy(2).build(signals, market_caps=caps).weights

    assert weights["weight"].to_list() == [pytest.approx(0.25), pytest.approx(0.75)]
    with pytest.raises(InputValidationError, match="requires market_caps"):
        FloatMarketCapWeightPolicy(2).build(signals)


def test_target_volatility_skips_snapshots_without_history() -> None:
    dates = [date(2024, 1, day) for day in range(1, 5)]
    signals = pl.DataFrame(
        {
            "time": [dates[0], dates[0], dates[3], dates[3]],
            "asset_id": ["a", "b", "a", "b"],
            "signal": [2.0, 1.0, 2.0, 1.0],
        }
    )
    prices = pl.DataFrame(
        {
            "time": dates * 2,
            "asset_id": ["a"] * 4 + ["b"] * 4,
            "price": [10.0, 11.0, 12.0, 13.0, 10.0, 9.0, 8.0, 7.0],
        }
    )
    policy = TargetVolatilityPolicy(EqualWeightPolicy(1), lookback_sessions=2)

    built = policy.build(
        signals, prices=prices, config=BacktestConfig(initial_capital=10_000)
    )

    assert built.skipped["reason"].to_list() == ["insufficient_volatility_history"]
    assert built.weights.get_column("weight").max() <= 1.0
