from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from bagelquant_core import Domain, IdentityPredictionComposer, Panel, PredictionPanel
from polars.testing import assert_frame_equal

from bagelquant_bt import (
    BacktestConfig,
    EqualWeightPolicy,
    FloatMarketCapWeightPolicy,
    ScheduledPrediction,
    TargetVolatilityPolicy,
    resolve_alpha_policy,
    resolve_execution_policy,
    resolve_standardize_policy,
    run_prediction_evaluation,
)
from bagelquant_bt.engine import run_weight_backtest
from bagelquant_bt.exceptions import InputValidationError


def _calendar() -> pl.DataFrame:
    return pl.DataFrame(
        {"time": pl.date_range(date(2024, 1, 1), date(2024, 3, 31), "1d", eager=True)}
    ).filter(pl.col("time").dt.weekday() <= 5)


def _signal_panel(
    frame: pl.DataFrame,
    calendar: pl.DataFrame | None = None,
) -> PredictionPanel:
    values = frame.rename(
        {"prediction": "value"}
        if "prediction" in frame.columns
        else {"signal": "value"}
    )
    dates = (
        calendar.get_column("time").cast(pl.Date).unique().sort()
        if calendar is not None
        else values.get_column("time").cast(pl.Date).unique().sort()
    )
    domain = Domain(
        calendar=dates,
        universe=values.get_column("asset_id").unique().sort(),
    )
    return PredictionPanel.from_domain(values, domain, name="signal")


def _scheduled_signal(frame: pl.DataFrame) -> ScheduledPrediction:
    panel = _signal_panel(frame)
    return ScheduledPrediction(
        schedule=frame.select("time").unique().sort("time"),
        prediction=panel,
    )


def test_prediction_executes_on_next_open_session() -> None:
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

    calendar = _calendar()
    scheduled = resolve_execution_policy("next_open").schedule_prediction(
        _signal_panel(predictions, calendar), calendar
    )
    assert scheduled.prediction.collect(dense=False).get_column(
        "time"
    ).unique().to_list() == [
        date(2024, 2, 1),
        date(2024, 3, 1),
    ]


def test_execution_uses_next_open_not_calendar_days() -> None:
    predictions = pl.DataFrame(
        {
            "time": [date(2024, 1, 31), date(2024, 1, 31)],
            "asset_id": ["a", "b"],
            "prediction": [2.0, 1.0],
        }
    )
    calendar = _calendar().filter(pl.col("time") != date(2024, 2, 1))

    scheduled = resolve_execution_policy("next_open").schedule_prediction(
        _signal_panel(predictions, calendar), calendar
    )

    assert scheduled.prediction.collect(dense=False).get_column(
        "time"
    ).unique().to_list() == [date(2024, 2, 2)]


def test_execution_policy_marks_rebalance_without_future_session() -> None:
    calendar = _calendar().filter(pl.col("time") <= date(2024, 2, 9))

    values = pl.DataFrame(
            {
                "time": [date(2024, 2, 9)],
                "asset_id": ["a"],
                "prediction": [1.0],
            }
        )
    selection = resolve_execution_policy("next_open").schedule_prediction(
        _signal_panel(values, calendar), calendar
    )

    assert selection.schedule.row(-1, named=True)["selection_status"] == "skipped"
    assert (
        selection.schedule.row(-1, named=True)["skip_reason"]
        == "missing_execution_session"
    )


def test_daily_policy_uses_exact_open_dates_and_next_session_execution() -> None:
    sessions = [date(2024, 1, 5), date(2024, 1, 9), date(2024, 1, 10)]
    calendar = pl.DataFrame({"time": sessions, "is_open": [1, 1, 1]})
    alpha = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [date(2024, 1, 5), date(2024, 1, 10)],
                "asset_id": ["a", "a"],
                "value": [1.0, 2.0],
            }
        ),
        Domain(calendar=sessions, universe=["a"]),
        name="alpha",
    )
    processed = resolve_alpha_policy("daily").apply({"alpha": alpha}, calendar)
    prediction = IdentityPredictionComposer().compose(
        processed.alpha_values["alpha"], name="prediction"
    ).compute()

    scheduled = resolve_execution_policy("next_open").schedule_prediction(
        prediction, calendar
    )

    assert processed.alpha_values["alpha"].collect(dense=False).get_column(
        "time"
    ).to_list() == [date(2024, 1, 5), date(2024, 1, 10)]
    assert scheduled.prediction.collect(dense=False).get_column("time").to_list() == [
        date(2024, 1, 9)
    ]
    assert scheduled.schedule.row(-1, named=True)["skip_reason"] == (
        "missing_execution_session"
    )


def test_alpha_policy_preserves_all_monthly_schedule_variants() -> None:
    calendar = pl.DataFrame(
        {
            "time": pl.date_range(date(2024, 1, 1), date(2024, 2, 5), "1d", eager=True)
        }
    ).with_columns((pl.col("time").dt.weekday() <= 5).cast(pl.Int8).alias("is_open"))

    mid = resolve_alpha_policy("monthly_mid").schedule(calendar)
    monday = resolve_alpha_policy("monthly_first_monday").schedule(calendar)
    friday = resolve_alpha_policy("monthly_last_friday").schedule(calendar)

    assert mid.row(0, named=True)["rebalance_date"] == date(2024, 1, 15)
    assert monday.row(0, named=True)["rebalance_date"] == date(2024, 1, 1)
    assert friday.row(0, named=True)["rebalance_date"] == date(2024, 1, 26)


def test_alpha_policy_aligns_evaluation_date_before_standardization() -> None:
    calendar = _calendar().filter(pl.col("time") <= date(2024, 1, 31))
    source_date = date(2024, 1, 30)
    domain = Domain(
        calendar=calendar.get_column("time"),
        universe=["a", "b"],
    )
    alpha = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [source_date, source_date],
                "asset_id": ["a", "b"],
                "value": [1.0, 3.0],
            }
        ),
        domain,
        name="alpha",
    )

    applied = resolve_standardize_policy("z_score").apply(
        resolve_alpha_policy("month_end").apply({"alpha": alpha}, calendar)
    )
    result = applied.alpha_values["alpha"].collect(dense=False)

    assert result.get_column("time").to_list() == [
        date(2024, 1, 31),
        date(2024, 1, 31),
    ]
    assert result.get_column("value").to_list() == pytest.approx(
        [-2**-0.5, 2**-0.5]
    )
    assert applied.alignments.to_dicts() == [
        {
            "alpha_name": "alpha",
            "observation_date": source_date,
            "evaluation_date": date(2024, 1, 31),
        }
    ]


def test_month_end_uses_latest_previous_snapshot_only_within_month() -> None:
    predictions = pl.DataFrame(
        {
            "time": [date(2024, 1, 30), date(2024, 1, 30), date(2024, 2, 1)],
            "asset_id": ["a", "b", "a"],
            "prediction": [2.0, 1.0, 99.0],
        }
    )

    calendar = _calendar()
    alpha = Panel.from_domain(
        predictions.rename({"prediction": "value"}),
        _signal_panel(predictions, calendar).domain,
        name="alpha",
    )
    processed = resolve_alpha_policy("month_end").apply(
        {"alpha": alpha}, calendar
    ).alpha_values["alpha"].collect(dense=False)
    january = processed.filter(pl.col("time") == date(2024, 1, 31))

    assert set(january["asset_id"]) == {
        "a",
        "b",
    }
    assert january.get_column("value").to_list() == [2.0, 1.0]


def test_month_end_skips_period_without_any_snapshot() -> None:
    predictions = pl.DataFrame(
        {
            "time": [date(2024, 1, 30)],
            "asset_id": ["a"],
            "prediction": [1.0],
        }
    )

    calendar = _calendar()
    alpha = Panel.from_domain(
        predictions.rename({"prediction": "value"}),
        _signal_panel(predictions, calendar).domain,
        name="alpha",
    )
    processed = resolve_alpha_policy("month_end").apply(
        {"alpha": alpha}, calendar
    ).alpha_values["alpha"].collect(dense=False)

    assert date(2024, 2, 29) not in processed.get_column("time")


def test_selection_identity_is_stable_for_all_consumers() -> None:
    predictions = pl.DataFrame(
        {
            "time": [date(2024, 1, 31), date(2024, 2, 29)],
            "asset_id": ["a", "a"],
            "prediction": [1.0, 2.0],
        }
    )
    calendar = _calendar()
    panel = _signal_panel(predictions, calendar)
    execution = resolve_execution_policy("next_open")
    first = execution.schedule_prediction(panel, calendar)
    second = execution.schedule_prediction(panel, calendar)

    assert first.identity == second.identity
    assert len(first.identity) == 32


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

    result = run_prediction_evaluation(
        _scheduled_signal(signals),
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
    assert result.quantile_returns["time"].unique().to_list() == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    assert result.quantile_returns.filter(
        pl.col("quantile") == "q1"
    )["return"].to_list() == [
        pytest.approx(0.1),
        pytest.approx(0.0909090909090908),
        pytest.approx(0.0),
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

    scheduled = _scheduled_signal(signals)
    weights = FloatMarketCapWeightPolicy(2).build(
        scheduled.prediction, market_caps=caps
    ).weights.collect(dense=False)

    assert weights["value"].to_list() == [pytest.approx(0.25), pytest.approx(0.75)]
    with pytest.raises(InputValidationError, match="requires market_caps"):
        FloatMarketCapWeightPolicy(2).build(scheduled)

    invalid_caps = caps.with_columns(
        pl.Series("float_market_cap", [-1.0, 1.0])
    )
    with pytest.raises(
        InputValidationError,
        match="float market caps must be finite and positive",
    ):
        FloatMarketCapWeightPolicy(2).build(
            scheduled.prediction,
            market_caps=invalid_caps,
        )


def test_factor_batch_preserves_custom_portfolio_policy_weights() -> None:
    times = [date(2024, 1, day) for day in range(1, 7)]
    signals = pl.DataFrame(
        {
            "time": [times[0]] * 3 + [times[3]] * 3,
            "asset_id": ["a", "b", "c"] * 2,
            "signal": [3.0, 2.0, 1.0, 2.0, 3.0, 1.0],
        }
    )
    prices = pl.DataFrame(
        {
            "time": times * 3,
            "asset_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
            "price": [
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                20.0,
                19.0,
                18.0,
                17.0,
                16.0,
                15.0,
                30.0,
                30.0,
                30.0,
                30.0,
                30.0,
                30.0,
            ],
        }
    )
    caps = pl.DataFrame(
        {
            "time": [times[0]] * 3 + [times[3]] * 3,
            "asset_id": ["a", "b", "c"] * 2,
            "float_market_cap": [1.0, 3.0, 2.0, 4.0, 1.0, 2.0],
        }
    )
    config = BacktestConfig(initial_capital=100_000, quantiles=2, top_n=2)
    policy = FloatMarketCapWeightPolicy(2)
    scheduled = _scheduled_signal(signals)
    expected_panel = policy.build(scheduled.prediction, market_caps=caps).weights
    expected_weights = expected_panel.collect(dense=False).rename({"value": "weight"})

    actual = run_prediction_evaluation(
        scheduled,
        prices,
        config=config,
        portfolio_policy=policy,
        portfolio_inputs={"market_caps": caps},
    )
    expected = run_weight_backtest(expected_weights, prices, config=config)

    assert actual.top_n_weights.equals(expected_weights)
    assert_frame_equal(
        actual.top_n_backtest.returns,
        expected.returns,
        check_exact=False,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


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
        _scheduled_signal(signals).prediction,
        prices=prices,
        config=BacktestConfig(initial_capital=10_000),
    )

    assert built.skipped["reason"].to_list() == ["insufficient_volatility_history"]
    assert built.weights.collect(dense=False).get_column("value").max() <= 1.0
