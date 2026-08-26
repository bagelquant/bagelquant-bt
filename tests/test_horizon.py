from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from bagelquant_core import Domain, PredictionPanel

from bagelquant_bt import (
    BacktestConfig,
    SessionWindow,
    TransactionCostConfig,
    benjamini_hochberg,
    centered_rank_book_weights,
    gross_one_tail_weights,
    hac_mean_test,
    implied_signal_half_life,
    non_overlapping_cohort_statistics,
    quantile_curve_structure,
    rolling_window_information_coefficients,
    run_daily_rank_path_diagnostics,
    run_prediction_horizon_diagnostics,
    session_window_forward_returns,
    window_book_returns,
    window_quantile_forward_returns,
    window_tail_returns,
)
from bagelquant_bt.policy import ScheduledPrediction


def _scheduled_prediction(
    *,
    evaluation_dates: list[date],
    execution_dates: list[date],
    values: dict[date, list[float]],
    assets: list[str],
) -> ScheduledPrediction:
    rows = [
        {"time": execution_date, "asset_id": asset, "value": signal}
        for execution_date in execution_dates
        for asset, signal in zip(assets, values[execution_date], strict=True)
    ]
    frame = pl.DataFrame(rows)
    domain = Domain(calendar=execution_dates, universe=assets)
    schedule = pl.DataFrame(
        {
            "requested_rebalance_date": evaluation_dates,
            "rebalance_date": evaluation_dates,
            "execution_date": execution_dates,
        },
        schema={
            "requested_rebalance_date": pl.Date,
            "rebalance_date": pl.Date,
            "execution_date": pl.Date,
        },
    )
    return ScheduledPrediction(
        schedule=schedule,
        prediction=PredictionPanel.from_domain(frame, domain, name="prediction"),
    )


def _factor(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "evaluation_date": [date(2024, 1, 1)] * len(values),
            "execution_date": [date(2024, 1, 2)] * len(values),
            "asset_id": [f"a{index}" for index in range(len(values))],
            "factor": values,
        }
    )


def _daily_scheduled_prediction(
    sessions: list[date],
    assets: list[str],
    *,
    reverse_every_other_day: bool = False,
) -> ScheduledPrediction:
    values = {}
    ascending = [float(index) for index in range(len(assets))]
    for index, session in enumerate(sessions[:-1]):
        values[session] = (
            list(reversed(ascending))
            if reverse_every_other_day and index % 2
            else ascending
        )
    return _scheduled_prediction(
        evaluation_dates=sessions[:-1],
        execution_dates=sessions[:-1],
        values=values,
        assets=assets,
    )


def test_centered_rank_book_weights_match_hand_calculation() -> None:
    weights = centered_rank_book_weights(_factor([1.0, 2.0, 3.0, 4.0]))

    assert weights.get_column("book_weight").to_list() == pytest.approx(
        [-0.375, -0.125, 0.125, 0.375]
    )
    assert weights.get_column("book_weight").sum() == pytest.approx(0.0)
    assert weights.filter(pl.col("book_weight") > 0).get_column(
        "book_weight"
    ).sum() == pytest.approx(0.5)
    assert weights.filter(pl.col("book_weight") < 0).get_column(
        "book_weight"
    ).sum() == pytest.approx(-0.5)
    assert weights.get_column("book_weight").abs().sum() == pytest.approx(1.0)


def test_centered_rank_book_uses_average_ties_and_rejects_constants() -> None:
    tied = centered_rank_book_weights(_factor([1.0, 2.0, 2.0, 3.0, 4.0]))
    tied_weights = tied.get_column("book_weight").to_list()

    assert tied_weights[1] == pytest.approx(tied_weights[2])
    assert sum(tied_weights) == pytest.approx(0.0)
    assert sum(abs(value) for value in tied_weights) == pytest.approx(1.0)

    constant = centered_rank_book_weights(_factor([2.0, 2.0, 2.0]))
    assert constant.get_column("book_weight").null_count() == 3
    assert constant.get_column("unavailable_reason").null_count() == 0


def test_centered_rank_book_handles_odd_and_sparse_dynamic_cross_sections() -> None:
    odd = centered_rank_book_weights(_factor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert odd.filter(pl.col("asset_id") == "a2").get_column(
        "book_weight"
    ).item() == pytest.approx(0.0)
    assert odd.get_column("book_weight").abs().sum() == pytest.approx(1.0)

    sparse = _factor([1.0, 2.0, float("nan"), 4.0, 5.0]).filter(
        pl.col("asset_id") != "a1"
    )
    weights = centered_rank_book_weights(sparse)
    assert weights.get_column("asset_id").to_list() == ["a0", "a3", "a4"]
    assert weights.get_column("book_weight").abs().sum() == pytest.approx(1.0)


def test_gross_one_tail_weights_use_half_books_and_zero_middle() -> None:
    weights = gross_one_tail_weights(_factor([float(value) for value in range(10)]))

    assert weights.get_column("tail_weight").abs().sum() == pytest.approx(1.0)
    assert weights.filter(pl.col("tail_weight") > 0).get_column(
        "tail_weight"
    ).sum() == pytest.approx(0.5)
    assert weights.filter(pl.col("tail_weight") < 0).get_column(
        "tail_weight"
    ).sum() == pytest.approx(-0.5)
    assert weights.filter(pl.col("tail_weight") == 0.0).height == 8


def test_session_windows_use_execution_price_and_bucket_offsets() -> None:
    sessions = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(9)]
    signals = _scheduled_prediction(
        evaluation_dates=[sessions[0], sessions[5]],
        execution_dates=[sessions[1], sessions[6]],
        values={sessions[1]: [1.0], sessions[6]: [2.0]},
        assets=["a"],
    )
    prices = pl.DataFrame(
        {
            "time": sessions,
            "asset_id": ["a"] * len(sessions),
            "price": [100.0 + 10.0 * offset for offset in range(len(sessions))],
        }
    )
    windows = (
        SessionWindow("cumulative", "cumulative_1d", 1, 1),
        SessionWindow("bucket", "bucket_2_5d", 2, 5),
    )

    forward, coverage = session_window_forward_returns(
        signals,
        prices,
        windows=windows,
        calendar=pl.DataFrame({"time": sessions}),
    )

    first = forward.filter(
        (pl.col("evaluation_date") == sessions[0])
        & (pl.col("window_id") == "cumulative_1d")
    ).row(0, named=True)
    assert first["execution_date"] == sessions[1]
    assert first["target_end_date"] == sessions[2]
    assert first["start_offset"] == 0
    assert first["end_offset"] == 1
    assert first["forward_return"] == pytest.approx(120.0 / 110.0 - 1.0)

    bucket = forward.filter(pl.col("window_id") == "bucket_2_5d").row(
        0, named=True
    )
    assert bucket["start_offset"] == 1
    assert bucket["end_offset"] == 5
    assert bucket["forward_return"] == pytest.approx(160.0 / 120.0 - 1.0)

    unfinished = coverage.filter(
        (pl.col("evaluation_date") == sessions[5])
        & (pl.col("window_id") == "bucket_2_5d")
    ).row(0, named=True)
    assert unfinished["target_available"] is False
    assert unfinished["coverage_ratio"] is None
    assert forward.filter(pl.col("evaluation_date") == sessions[5]).height == 1


def test_session_window_price_gap_freezes_then_recognizes_cumulative_move() -> None:
    sessions = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(5)]
    signals = _scheduled_prediction(
        evaluation_dates=[sessions[0]],
        execution_dates=[sessions[1]],
        values={sessions[1]: [1.0]},
        assets=["a"],
    )
    prices = pl.DataFrame(
        {
            "time": [sessions[1], sessions[4]],
            "asset_id": ["a", "a"],
            "price": [100.0, 130.0],
        }
    )
    windows = (
        SessionWindow("cumulative", "cumulative_1d", 1, 1),
        SessionWindow("cumulative", "cumulative_3d", 1, 3),
    )

    forward, _ = session_window_forward_returns(
        signals,
        prices,
        windows=windows,
        calendar=pl.DataFrame({"time": sessions}),
    )

    returns = dict(
        forward.select("window_id", "forward_return").iter_rows()
    )
    assert returns["cumulative_1d"] == pytest.approx(0.0)
    assert returns["cumulative_3d"] == pytest.approx(0.3)


def test_book_return_never_reweights_around_missing_forward_labels() -> None:
    factor = _factor([1.0, 2.0, 3.0, 4.0])
    weights = centered_rank_book_weights(factor)
    forward = pl.DataFrame(
        {
            "evaluation_date": [date(2024, 1, 1)] * 4,
            "execution_date": [date(2024, 1, 2)] * 4,
            "target_end_date": [date(2024, 1, 3)] * 4,
            "asset_id": ["a0", "a1", "a2", "a3"],
            "window_kind": ["cumulative"] * 4,
            "window_id": ["cumulative_1d"] * 4,
            "start_session": [1] * 4,
            "end_session": [1] * 4,
            "start_offset": [0] * 4,
            "end_offset": [1] * 4,
            "forward_return": [-0.03, -0.01, 0.01, 0.03],
        }
    )

    complete = window_book_returns(weights, forward).row(0, named=True)
    assert complete["book_return"] == pytest.approx(0.025)
    assert complete["coverage_ratio"] == pytest.approx(1.0)

    missing = window_book_returns(
        weights,
        forward.with_columns(
            pl.when(pl.col("asset_id") == "a3")
            .then(None)
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        ),
    ).row(0, named=True)
    assert missing["book_return"] is None
    assert missing["coverage_ratio"] == pytest.approx(0.75)


def test_book_tail_and_quantiles_reveal_different_cross_section_structure() -> None:
    factor = _factor([float(value) for value in range(10)])
    metadata = {
        "evaluation_date": [date(2024, 1, 1)] * 10,
        "execution_date": [date(2024, 1, 2)] * 10,
        "target_end_date": [date(2024, 1, 3)] * 10,
        "asset_id": [f"a{index}" for index in range(10)],
        "window_kind": ["cumulative"] * 10,
        "window_id": ["cumulative_1d"] * 10,
        "start_session": [1] * 10,
        "end_session": [1] * 10,
        "start_offset": [0] * 10,
        "end_offset": [1] * 10,
    }
    book_weights = centered_rank_book_weights(factor)
    tail_weights = gross_one_tail_weights(factor)

    middle_only = pl.DataFrame(
        {
            **metadata,
            "forward_return": [0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
        }
    )
    assert window_book_returns(book_weights, middle_only).get_column(
        "book_return"
    ).item() > 0.0
    assert window_tail_returns(tail_weights, middle_only).get_column(
        "tail_return"
    ).item() == pytest.approx(0.0)

    extreme_only = pl.DataFrame(
        {**metadata, "forward_return": [-0.1] + [0.0] * 8 + [0.1]}
    )
    book = window_book_returns(book_weights, extreme_only).get_column(
        "book_return"
    ).item()
    tail = window_tail_returns(tail_weights, extreme_only).get_column(
        "tail_return"
    ).item()
    assert 0.0 < book < tail

    u_shape = pl.DataFrame(
        {**metadata, "forward_return": [0.1] + [0.0] * 8 + [0.1]}
    )
    assert window_book_returns(book_weights, u_shape).get_column(
        "book_return"
    ).item() == pytest.approx(0.0)
    quantiles = window_quantile_forward_returns(factor, u_shape)
    endpoints = quantiles.filter(pl.col("quantile").is_in(["q1", "q10"]))
    assert endpoints.get_column("quantile_return").to_list() == pytest.approx(
        [0.1, 0.1]
    )

    flat = window_quantile_forward_returns(
        factor,
        pl.DataFrame({**metadata, "forward_return": [0.0] * 10}),
    )
    flat_structure = quantile_curve_structure(flat)
    assert flat_structure.get_column("quantile_rank_ic").item() is None
    assert flat_structure.get_column("unavailable_reason").item() == (
        "quantile-rank IC requires non-constant returns"
    )


def test_hac_bh_and_all_staggered_cohorts_are_deterministic() -> None:
    values = [0.01, 0.02, -0.01, 0.03, 0.02, 0.04]
    result = hac_mean_test(values, window_width=2)

    assert result.sample_size == 6
    assert result.lag == 2
    assert result.mean == pytest.approx(sum(values) / len(values))
    assert result.standard_error is not None
    assert result.confidence_low < result.mean < result.confidence_high

    q_values = benjamini_hochberg([0.01, 0.04, 0.03, None])
    assert q_values == pytest.approx([0.03, 0.04, 0.04, None])

    dates = [date(2024, 1, 1) + timedelta(days=value) for value in range(6)]
    cohorts = non_overlapping_cohort_statistics(
        dates, values, window_width=2
    )
    assert cohorts["cohort_count"] == 2
    assert cohorts["cohort_same_sign_ratio"] == pytest.approx(1.0)


def test_complete_protocol_keeps_horizons_and_persistence_explicit() -> None:
    sessions = [date(2024, 1, 1) + timedelta(days=value) for value in range(131)]
    assets = [f"a{value:02d}" for value in range(10)]
    evaluation_dates = sessions[:-2]
    execution_dates = sessions[1:-1]
    values = {
        execution_date: [float(rank) for rank in range(10)]
        for execution_date in execution_dates
    }
    signals = _scheduled_prediction(
        evaluation_dates=evaluation_dates,
        execution_dates=execution_dates,
        values=values,
        assets=assets,
    )
    prices = pl.DataFrame(
        [
            {
                "time": current_date,
                "asset_id": asset,
                "price": 100.0 * (1.0 + 0.0001 * rank) ** session,
            }
            for session, current_date in enumerate(sessions)
            for rank, asset in enumerate(assets)
        ]
    )

    result = run_prediction_horizon_diagnostics(
        signals,
        prices,
        calendar=pl.DataFrame({"time": sessions}),
    )

    assert result.ic_summary.get_column("window_id").n_unique() == 12
    assert result.book_returns.get_column("window_id").n_unique() == 12
    assert result.tail_returns.get_column("window_id").n_unique() == 12
    assert result.quantile_forward_returns.get_column("quantile").n_unique() == 10
    assert result.signal_persistence_summary.get_column(
        "signal_half_life_band"
    ).unique().to_list() == [">120D"]
    assert set(result.statistical_inference.get_column("metric")) == {
        "book_return",
        "cross_section_regression",
        "pearson_ic",
        "quantile_rank_ic",
        "spearman_ic",
        "tail_return",
    }


def test_daily_rank_paths_apply_costs_and_distinguish_requested_execution() -> None:
    sessions = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(4)]
    assets = [f"a{index}" for index in range(10)]
    signals = _daily_scheduled_prediction(sessions, assets)
    prices = pl.DataFrame(
        {
            "time": [session for session in sessions for _ in assets],
            "asset_id": assets * len(sessions),
            "price": [
                100.0 + index * day
                for day, _session in enumerate(sessions)
                for index in range(len(assets))
            ],
        }
    )
    availability = pl.DataFrame(
        {
            "time": [sessions[0]],
            "asset_id": [assets[-1]],
            "can_buy": [False],
            "can_sell": [True],
            "reason": ["blocked_initial_long"],
        }
    )
    result = run_daily_rank_path_diagnostics(
        signals,
        prices,
        config=BacktestConfig(
            initial_capital=1_000_000.0,
            annualization=240,
            transaction_cost=TransactionCostConfig(
                rate=0.001,
                min_fee=0.0,
                buy_slippage_rate=0.0,
                sell_slippage_rate=0.0,
                stamp_tax_rate=0.0,
            ),
        ),
        execution_availability=availability,
        calendar=pl.DataFrame({"time": sessions}),
        lead_lags=(-1, 0, 1),
        autocorrelation_lags=(1, 2),
        rolling_observations=2,
    )

    first_turnover = result.book_turnover.row(0, named=True)
    assert first_turnover["requested_turnover"] == pytest.approx(1.0)
    assert first_turnover["executed_turnover"] < first_turnover["requested_turnover"]
    assert first_turnover["is_initial_rebalance"] is True
    assert result.book_turnover.get_column("is_initial_rebalance").sum() == 1
    first_return = result.book_daily_returns.row(0, named=True)
    assert first_return["net_return"] < first_return["gross_return"]
    assert set(result.tail_daily_returns.columns) == {
        "time",
        "gross_return",
        "net_return",
    }


def test_daily_lead_lag_has_every_integer_offset_on_one_common_sample() -> None:
    sessions = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(92)]
    assets = [f"a{index}" for index in range(10)]
    signals = _daily_scheduled_prediction(
        sessions,
        assets,
        reverse_every_other_day=True,
    )
    prices = pl.DataFrame(
        {
            "time": [session for session in sessions for _ in assets],
            "asset_id": assets * len(sessions),
            "price": [100.0] * (len(sessions) * len(assets)),
        }
    )
    progress: list[tuple[str, int, int]] = []
    result = run_daily_rank_path_diagnostics(
        signals,
        prices,
        config=BacktestConfig(initial_capital=1_000_000.0, annualization=240),
        calendar=pl.DataFrame({"time": sessions}),
        progress=lambda label, completed, total: progress.append(
            (label, completed, total)
        ),
    )

    lead_lag = result.book_lead_lag_returns
    assert lead_lag.get_column("lag").unique().sort().to_list() == list(
        range(-30, 31)
    )
    sample_sizes = lead_lag.group_by("lag").len().get_column("len").to_list()
    assert len(set(sample_sizes)) == 1
    date_counts = (
        lead_lag.group_by("lag")
        .agg(pl.col("time").n_unique().alias("dates"))
        .get_column("dates")
        .to_list()
    )
    assert date_counts == sample_sizes
    alpha_return = result.alpha_return_lag_returns
    assert set(alpha_return.get_column("path_kind")) == {"book", "tail"}
    assert alpha_return.get_column("lag").unique().sort().to_list() == [
        0,
        1,
        2,
        5,
        10,
        20,
        60,
    ]
    alpha_sample_sizes = (
        alpha_return.group_by("path_kind", "lag").len().get_column("len").to_list()
    )
    assert len(set(alpha_sample_sizes)) == 1
    assert ("book_tail_paths", 1, 1) in progress
    assert ("book_lead_lag_paths", 61, 61) in progress
    assert ("alpha_return_lag_paths", 14, 14) in progress
    assert ("signal_autocorrelation", 120, 120) in progress


def test_daily_summary_autocorrelation_grid_and_half_life_formula() -> None:
    sessions = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(125)]
    assets = [f"a{index}" for index in range(4)]
    signals = _daily_scheduled_prediction(sessions, assets)
    prices = pl.DataFrame(
        {
            "time": [session for session in sessions for _ in assets],
            "asset_id": assets * len(sessions),
            "price": [100.0] * (len(sessions) * len(assets)),
        }
    )
    result = run_daily_rank_path_diagnostics(
        signals,
        prices,
        config=BacktestConfig(initial_capital=1_000_000.0, annualization=240),
        calendar=pl.DataFrame({"time": sessions}),
    )

    assert result.signal_autocorrelation.get_column(
        "horizon_sessions"
    ).unique().sort().to_list() == list(range(1, 121))
    assert result.signal_autocorrelation.get_column(
        "rank_autocorrelation"
    ).drop_nulls().to_list() == pytest.approx(
        [1.0] * result.signal_autocorrelation.height
    )
    assert implied_signal_half_life(5, 0.5) == pytest.approx(5.0)
    assert implied_signal_half_life(5, 0.0) is None
    assert implied_signal_half_life(5, 1.0) is None
    assert implied_signal_half_life(5, -0.5) is None


def test_rolling_ic_uses_240_valid_causal_observations() -> None:
    sessions = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(242)]
    pearson: list[float | None] = [float(index) for index in range(242)]
    pearson[100] = None
    ic = pl.DataFrame(
        {
            "evaluation_date": sessions,
            "window_kind": ["cumulative"] * len(sessions),
            "window_id": ["cumulative_1d"] * len(sessions),
            "start_session": [1] * len(sessions),
            "end_session": [1] * len(sessions),
            "pearson_ic": pearson,
            "spearman_ic": [0.25] * len(sessions),
        }
    )

    rolling = rolling_window_information_coefficients(ic)
    pearson_rolling = rolling.filter(pl.col("method") == "pearson").sort(
        "evaluation_date"
    )

    non_null = pearson_rolling.filter(pl.col("rolling_ic").is_not_null())
    assert non_null.height == 2
    first = non_null.row(0, named=True)
    assert first["evaluation_date"] == sessions[240]
    expected = [float(index) for index in range(241) if index != 100]
    assert first["rolling_ic"] == pytest.approx(sum(expected) / 240.0)
    assert first["rolling_observations"] == 240
