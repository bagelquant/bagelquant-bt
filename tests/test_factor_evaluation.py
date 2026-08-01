from __future__ import annotations

import math
from dataclasses import asdict
from datetime import date

import polars as pl
import pytest
from bagelquant_core import Domain, SignalPanel
from polars.testing import assert_frame_equal

import bagelquant_bt.factor as factor_module
from bagelquant_bt import (
    BacktestConfig,
    ScheduledSignal,
    materialize_signal_diagnostics,
    run_signal_evaluation,
)
from bagelquant_bt.engine import (
    _legacy_compact_backtest_weight_frame_with_active_market,
    _run_sparse_compact_backtests,
)
from bagelquant_bt.exceptions import InputValidationError
from bagelquant_bt.factor import (
    _traded_factor_quantile_returns_with_forward_returns,
    factor_ic_decay,
    factor_quantile_returns,
    information_coefficients,
    lag_factor,
    prepare_factor_market_data,
    quantile_equal_weights,
    run_factor_evaluation,
    signal_forward_returns,
    spread_quantile_weights,
    summarize_ic,
)


def _scheduled_signal(frame: pl.DataFrame) -> ScheduledSignal:
    values = frame.with_columns(pl.col("time").cast(pl.Date)).rename(
        {"signal": "value"}
    )
    domain = Domain(
        calendar=values.get_column("time").unique().sort(),
        universe=values.get_column("asset_id").unique().sort(),
    )
    return ScheduledSignal(
        schedule=values.select("time").unique().sort("time"),
        signal=SignalPanel.from_domain(values, domain, name="signal"),
    )


def test_icir_is_annualized_with_configured_ic_observations() -> None:
    ic = pl.DataFrame(
        {
            "pearson_ic": [0.1, 0.2, 0.3],
            "spearman_ic": [0.3, 0.2, 0.1],
        }
    )

    summary = summarize_ic(ic, annualization=240)

    expected = 2.0 * math.sqrt(240)
    pearson_icir = summary.filter(pl.col("method") == "pearson").item(
        0, "icir"
    )
    spearman_icir = summary.filter(pl.col("method") == "spearman").item(
        0, "icir"
    )
    assert pearson_icir == pytest.approx(expected)
    assert spearman_icir == pytest.approx(expected)


def test_factor_coverage_uses_explicit_membership_universe() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
            "asset_id": ["a", "b", "c", "d"] * 2,
            "price": [1.0, 2.0, 3.0, 4.0, 1.1, 2.1, 3.1, 4.1],
        }
    )
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "factor": [1.0, 2.0],
        }
    )
    membership = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "asset_id": ["a", "b", "a"],
        }
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
        coverage_universe=membership,
    )

    assert result.coverage.to_dicts() == [
        {
            "time": date(2024, 1, 1),
            "factor_signal_asset_count": 2,
            "universe_asset_count": 2,
            "coverage_ratio": 1.0,
        },
        {
            "time": date(2024, 1, 2),
            "factor_signal_asset_count": 0,
            "universe_asset_count": 1,
            "coverage_ratio": 0.0,
        },
    ]


def test_factor_evaluation_uses_time_asset_id_inputs() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "asset_id": ["a", "a", "b", "b"],
            "price": [1.0, 2.0, 2.0, 1.0],
        }
    )
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "factor": [2.0, 1.0],
        }
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )

    assert result.ic.height == 1
    assert result.ic.columns == ["time", "pearson_ic", "spearman_ic"]
    assert result.ic_summary["method"].to_list() == ["pearson", "spearman"]
    assert result.quantile_returns.select("time", "quantile", "return").height == 2
    assert not hasattr(result, "top_minus_bottom")
    assert not hasattr(result, "long_short_weights")
    assert not hasattr(result, "long_short_backtest")
    assert result.top_n_weights.to_dicts()[0]["asset_id"] == "a"
    assert result.top_n_backtest.transaction_costs.data["total_fee"].sum() > 0
    assert (
        result.top_n_backtest.performance.filter(pl.col("metric") == "sharpe").height
        == 1
    )


@pytest.mark.parametrize("retry_blocked", [True, False])
def test_batched_quantiles_match_sequential_portfolios(
    retry_blocked: bool,
) -> None:
    times = [date(2024, 1, day) for day in range(1, 9)]
    assets = ["a", "b", "c", "d"]
    prices = pl.DataFrame(
        {
            "time": [time for asset in assets for time in times],
            "asset_id": [asset for asset in assets for _ in times],
            "price": [
                100.0 + asset_index * 3.0 + day_index * (asset_index - 1.5)
                for asset_index, _ in enumerate(assets)
                for day_index, _ in enumerate(times)
            ],
        }
    )
    factor = pl.DataFrame(
        {
            "time": [times[0]] * 4 + [times[3]] * 4,
            "asset_id": assets * 2,
            "factor": [4.0, 3.0, 2.0, 1.0, 1.0, 4.0, 2.0, 3.0],
        }
    )
    availability = pl.DataFrame(
        {
            "time": [times[0], times[1], times[3], times[4]],
            "asset_id": ["a", "a", "d", "d"],
            "can_buy": [False, False, True, True],
            "can_sell": [True, True, False, False],
            "reason": ["limit"] * 4,
        }
    )
    config = BacktestConfig(
        initial_capital=100_000,
        quantiles=2,
        top_n=1,
        retry_blocked_orders=retry_blocked,
    )
    market = prepare_factor_market_data(prices)
    actual = _traded_factor_quantile_returns_with_forward_returns(
        factor,
        market.prices,
        config=config,
        quantiles=2,
        forward_returns=market.forward_returns,
        price_gaps=market.price_data.price_gaps if market.price_data else None,
        execution_availability=availability,
    ).select("time", "quantile", "return")

    expected_frames = []
    weight_frames = quantile_equal_weights(factor, quantiles=2)
    batched = _run_sparse_compact_backtests(
        weight_frames,
        market.prices,
        market.forward_returns,
        config=config,
        execution_availability=availability,
        execution_availability_validated=False,
    )
    for label, weights in weight_frames.items():
        backtest = _legacy_compact_backtest_weight_frame_with_active_market(
            weights,
            market.prices,
            market.forward_returns,
            config=config,
            execution_availability=availability,
            execution_availability_validated=False,
        )
        assert_frame_equal(
            batched[label].returns,
            backtest.returns,
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        assert_frame_equal(
            batched[label].value,
            backtest.value,
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for field, expected_value in asdict(backtest.summary).items():
            assert getattr(batched[label].summary, field) == pytest.approx(
                expected_value, rel=1e-12, abs=1e-12, nan_ok=True
            )
        expected_frames.append(
            backtest.returns.select(
                "time",
                pl.lit(label).alias("quantile"),
                pl.col("gross_return").alias("return"),
            )
        )
    expected = pl.concat(expected_frames).sort(["time", "quantile"])
    assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    evaluation = run_factor_evaluation(
        factor,
        prices,
        config=config,
        market_data=market,
        execution_availability=availability,
    )
    assert_frame_equal(
        evaluation.quantile_returns.select("time", "quantile", "return"),
        expected,
        check_exact=False,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_sparse_batch_matches_reference_across_listing_and_price_gaps() -> None:
    times = [date(2024, 1, day) for day in range(1, 7)]
    prices = pl.DataFrame(
        {
            "time": [
                *times,
                times[1],
                times[3],
                times[4],
                times[5],
                *times[:4],
            ],
            "asset_id": [*["a"] * 6, *["b"] * 4, *["c"] * 4],
            "price": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                50.0,
                55.0,
                56.0,
                57.0,
                80.0,
                79.0,
                78.0,
                77.0,
            ],
        }
    )
    weight_frames = {
        "gap_exit": pl.DataFrame(
            {
                "time": [times[0], times[1], times[2], times[4]],
                "asset_id": ["a", "b", "a", "b"],
                "weight": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        "delisting": pl.DataFrame(
            {
                "time": [times[0], times[2], times[4]],
                "asset_id": ["c", "a", "a"],
                "weight": [1.0, 1.0, 1.0],
            }
        ),
    }
    market = prepare_factor_market_data(prices)
    config = BacktestConfig(initial_capital=100_000)

    actual = _run_sparse_compact_backtests(
        weight_frames,
        market.prices,
        market.forward_returns,
        config=config,
        execution_availability=None,
        execution_availability_validated=True,
    )

    for label, weights in weight_frames.items():
        expected = _legacy_compact_backtest_weight_frame_with_active_market(
            weights,
            market.prices,
            market.forward_returns,
            config=config,
            execution_availability=None,
            execution_availability_validated=True,
        )
        for actual_frame, expected_frame in (
            (actual[label].returns, expected.returns),
            (actual[label].value, expected.value),
            (actual[label].turnover, expected.turnover),
            (actual[label].costs.data, expected.costs.data),
        ):
            assert_frame_equal(
                actual_frame,
                expected_frame,
                check_exact=False,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
def test_prepared_signal_returns_validate_schedule_and_price_keys() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-02-01"] * 2,
            "asset_id": ["a", "a", "b", "b"],
            "price": [1.0, 1.1, 2.0, 2.1],
        }
    )
    signals = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "signal": [2.0, 1.0],
        }
    )
    config = BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1)

    with pytest.raises(InputValidationError, match="outside the current signal"):
        run_signal_evaluation(
            _scheduled_signal(signals),
            prices,
            config=config,
            evaluation_returns=pl.DataFrame(
                {
                    "time": ["2024-02-01"],
                    "asset_id": ["a"],
                    "forward_return": [0.1],
                }
            ),
        )

    with pytest.raises(InputValidationError, match="outside the current signals"):
        run_signal_evaluation(
            _scheduled_signal(signals),
            prices,
            config=config,
            evaluation_returns=pl.DataFrame(
                {
                    "time": ["2024-01-01"],
                    "asset_id": ["missing"],
                    "forward_return": [0.1],
                }
            ),
        )

    signals_with_missing_price = pl.concat(
        [
            signals,
            pl.DataFrame(
                {
                    "time": ["2024-01-01"],
                    "asset_id": ["missing"],
                    "signal": [0.0],
                }
            ),
        ]
    )
    with pytest.raises(InputValidationError, match="absent from prepared prices"):
        run_signal_evaluation(
            _scheduled_signal(signals_with_missing_price),
            prices,
            config=config,
            evaluation_returns=pl.DataFrame(
                {
                    "time": ["2024-01-01"],
                    "asset_id": ["missing"],
                    "forward_return": [0.1],
                }
            ),
        )


def test_factor_evaluation_adds_spread_and_lag_outputs() -> None:
    dates = (
        [f"2024-01-{day:02d}" for day in range(1, 29)]
        + [f"2024-02-{day:02d}" for day in range(1, 29)]
        + [f"2024-03-{day:02d}" for day in range(1, 11)]
    )
    assets = ["a", "b", "c", "d"]
    prices = pl.DataFrame(
        {
            "time": [date for asset in assets for date in dates],
            "asset_id": [asset for asset in assets for _ in dates],
            "price": [
                10.0 + index * (0.2 if asset in {"a", "c"} else -0.1)
                for asset in assets
                for index, _ in enumerate(dates)
            ],
        }
    )
    factor = prices.select("time", "asset_id").with_columns(
        pl.when(pl.col("asset_id").is_in(["a", "c"]))
        .then(2.0)
        .otherwise(1.0)
        .alias("factor")
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )

    assert result.spread_weights.height > 0
    assert result.spread_backtest is not None
    assert result.spread_backtest.transaction_costs.data["total_fee"].sum() > 0
    assert set(result.lag_analysis["portfolio"]) == {"top_n", "spread"}
    expected_lags = {0, 1, 2, 3, 4, 5, 10, 20, 30, 60}
    assert set(result.lag_analysis["lag"]) == expected_lags
    assert set(result.lag_returns["lag"]) == expected_lags
    assert set(result.lag_returns["portfolio"]) == {"top_n", "spread"}
    assert set(result.ic_decay["method"]) == {"pearson", "spearman"}
    assert result.lag_analysis.select("portfolio", "lag").to_dicts() == sorted(
        result.lag_analysis.select("portfolio", "lag").to_dicts(),
        key=lambda row: (row["portfolio"], row["lag"]),
    )
    for row in result.lag_analysis.iter_rows(named=True):
        returns = result.lag_returns.filter(
            (pl.col("portfolio") == row["portfolio"]) & (pl.col("lag") == row["lag"])
        )
        if returns.is_empty():
            continue
        last = returns.tail(1).to_dicts()[0]
        assert last["gross_cumulative_return"] == pytest.approx(
            row["gross_cumulative_return"]
        )
        assert last["net_cumulative_return"] == pytest.approx(
            row["net_cumulative_return"]
        )


def test_sparse_factor_keeps_analytics_and_trades_daily_portfolios() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                date
                for asset in ["a", "b"]
                for date in [
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ],
            "asset_id": [asset for asset in ["a", "b"] for _ in range(4)],
            "price": [
                10.0,
                11.0,
                12.0,
                13.0,
                10.0,
                9.0,
                8.0,
                7.0,
            ],
        }
    )
    factor = pl.DataFrame(
        {
            "time": ["2024-01-03", "2024-01-03"],
            "asset_id": ["a", "b"],
            "factor": [2.0, 1.0],
        }
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )

    assert result.factor["time"].dt.strftime("%Y-%m-%d").unique().to_list() == [
        "2024-01-03"
    ]
    assert result.ic.height == 1
    assert result.icir == result.icir or result.ic_std != result.ic_std
    assert set(result.ic_decay["method"]) == {"pearson", "spearman"}
    assert result.top_n_backtest.returns["time"].dt.strftime("%Y-%m-%d").to_list() == [
        "2024-01-03",
        "2024-01-04",
    ]
    assert set(result.quantile_returns["quantile"]) == {"q1", "q2"}
    assert result.spread_backtest is not None
    assert result.top_n_backtest.transaction_costs.data["total_fee"].to_list()[1] == 0.0


def test_lag_factor_counts_daily_price_sessions_for_monthly_signals() -> None:
    sessions = pl.DataFrame(
        {
            "time": pl.date_range(
                date(2024, 1, 1), date(2024, 3, 31), "1d", eager=True
            )
        }
    ).filter(pl.col("time").dt.weekday() <= 5)
    sessions = sessions.filter(pl.col("time") != date(2024, 2, 12))
    factor = pl.DataFrame(
        {
            "time": [date(2024, 1, 31), date(2024, 1, 31)],
            "asset_id": ["a", "b"],
            "factor": [2.0, 1.0],
        }
    )

    lagged = lag_factor(factor, lag=15, trading_sessions=sessions)

    assert lagged.to_dicts() == [
        {"time": date(2024, 2, 22), "asset_id": "a", "factor": 2.0},
        {"time": date(2024, 2, 22), "asset_id": "b", "factor": 1.0},
    ]


def test_lag_factor_matches_observation_shift_for_daily_inputs() -> None:
    sessions = pl.DataFrame(
        {"time": [date(2024, 1, day) for day in range(2, 7)]}
    )
    factor = pl.DataFrame(
        {
            "time": [date(2024, 1, day) for day in range(2, 7)],
            "asset_id": ["a"] * 5,
            "factor": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    lagged = lag_factor(factor, lag=2, trading_sessions=sessions)

    assert lagged.to_dicts() == [
        {"time": date(2024, 1, 4), "asset_id": "a", "factor": 10.0},
        {"time": date(2024, 1, 5), "asset_id": "a", "factor": 20.0},
        {"time": date(2024, 1, 6), "asset_id": "a", "factor": 30.0},
    ]


def test_batched_ic_decay_matches_sequential_reference_with_missing_returns() -> None:
    sessions = pl.DataFrame(
        {"time": [date(2024, 1, day) for day in range(1, 7)]}
    )
    factor = pl.DataFrame(
        {
            "time": [date(2024, 1, 1)] * 4 + [date(2024, 1, 3)] * 4,
            "asset_id": ["a", "b", "c", "d"] * 2,
            "factor": [4.0, 3.0, 2.0, 1.0, 1.0, 3.0, 4.0, 2.0],
        }
    )
    forward_returns = pl.DataFrame(
        {
            "time": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 3),
                date(2024, 1, 3),
            ],
            "asset_id": ["a", "b", "d", "a", "b", "c", "d", "a", "c", "d"],
            "forward_return": [
                0.04,
                0.03,
                0.01,
                0.01,
                0.04,
                0.02,
                0.03,
                0.02,
                0.05,
                0.01,
            ],
        }
    )
    lags = (0, 1, 2)

    actual = factor_ic_decay(
        factor,
        forward_returns,
        trading_sessions=sessions,
        lags=lags,
    )
    expected = factor_ic_decay(
        factor,
        forward_returns,
        trading_sessions=sessions,
        return_provider=lambda _: forward_returns,
        lags=lags,
    )

    assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_monthly_signal_lag_trades_daily_from_the_shifted_session() -> None:
    sessions = pl.DataFrame(
        {
            "time": pl.date_range(
                date(2024, 1, 1), date(2024, 3, 31), "1d", eager=True
            )
        }
    ).filter(pl.col("time").dt.weekday() <= 5)
    sessions = sessions.filter(pl.col("time") != date(2024, 2, 12))
    prices = (
        sessions.join(pl.DataFrame({"asset_id": ["a", "b"]}), how="cross")
        .with_columns(
            pl.when(pl.col("asset_id") == "a")
            .then(10.0 + pl.int_range(0, pl.len()).over("asset_id") * 0.1)
            .otherwise(10.0 - pl.int_range(0, pl.len()).over("asset_id") * 0.1)
            .alias("price")
        )
    )
    signals = pl.DataFrame(
        {
            "time": [
                date(2024, 1, 31),
                date(2024, 1, 31),
                date(2024, 2, 29),
                date(2024, 2, 29),
            ],
            "asset_id": ["a", "b", "a", "b"],
            "signal": [2.0, 1.0, 1.0, 2.0],
        }
    )

    result = run_signal_evaluation(
        _scheduled_signal(signals),
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )
    delayed_returns = result.lag_returns.filter(
        (pl.col("lag") == 20) & (pl.col("portfolio") == "top_n")
    )

    assert delayed_returns["time"].head(2).to_list() == [
        date(2024, 2, 29),
        date(2024, 3, 1),
    ]
    assert result.ic_decay.filter(pl.col("lag") == 20)["ic_mean"].null_count() == 0


def test_signal_evaluation_reuses_prepared_prices_and_scheduled_returns() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-02-01", "2024-03-01"] * 2,
            "asset_id": ["a"] * 3 + ["b"] * 3,
            "price": [10.0, 11.0, 12.0, 20.0, 18.0, 21.0],
        }
    )
    signals = pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-02", "2024-02-01", "2024-02-01"],
            "asset_id": ["a", "b", "a", "b"],
            "signal": [2.0, 1.0, 1.0, 2.0],
        }
    )
    prepared = prepare_factor_market_data(prices)
    scheduled = signal_forward_returns(
        signals.select(
            "time", "asset_id", pl.col("signal").alias("factor")
        ).with_columns(pl.col("time").str.to_date()),
        prepared.prices,
    )

    direct = run_signal_evaluation(
        _scheduled_signal(signals),
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )
    reused = run_signal_evaluation(
        _scheduled_signal(signals),
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
        market_data=prepared,
        evaluation_returns=scheduled,
    )

    assert reused.ic.equals(direct.ic)
    assert reused.ic_summary.equals(direct.ic_summary)
    assert reused.quantile_returns.equals(direct.quantile_returns)
    assert reused.lag_analysis.equals(direct.lag_analysis)
    assert reused.lag_returns.equals(direct.lag_returns)


def test_factor_quantile_returns_preserve_bucket_semantics_and_low_counts() -> None:
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 4 + ["2024-01-02"],
            "asset_id": ["a", "b", "c", "d", "a"],
            "factor": [1.0, 2.0, 3.0, 4.0, 1.0],
        }
    )
    forward_returns = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 4 + ["2024-01-02"],
            "asset_id": ["a", "b", "c", "d", "a"],
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.10],
        }
    )

    result = factor_quantile_returns(factor, forward_returns, quantiles=2)

    returns = result.select("time", "quantile", "return").to_dicts()
    assert returns == [
        {"time": returns[0]["time"], "quantile": "q1", "return": pytest.approx(0.035)},
        {"time": returns[1]["time"], "quantile": "q2", "return": pytest.approx(0.015)},
        {"time": returns[2]["time"], "quantile": "q1", "return": None},
        {"time": returns[3]["time"], "quantile": "q2", "return": None},
    ]


def test_spread_is_q1_minus_qn_and_matches_the_spread_backtest() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"] * 2,
            "asset_id": ["high", "high", "low", "low"],
            "price": [100.0, 100.0, 100.0, 110.0],
        }
    )
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["high", "low"],
            "factor": [2.0, 1.0],
        }
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=1_000_000, quantiles=2, top_n=1),
    )

    assert result.quantile_returns.select("quantile", "return").to_dicts() == [
        {"quantile": "q1", "return": pytest.approx(0.0)},
        {"quantile": "q2", "return": pytest.approx(0.1)},
    ]
    assert result.spread_returns["spread_return"].to_list() == [pytest.approx(-0.1)]
    assert result.spread_backtest is not None
    assert result.spread_backtest.returns["gross_return"].to_list() == [
        pytest.approx(-0.1)
    ]


def test_information_coefficients_keep_null_rows_for_unusable_dates() -> None:
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "factor": [None, None],
        }
    )
    forward_returns = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "forward_return": [0.01, 0.02],
        }
    )

    result = information_coefficients(factor, forward_returns)

    assert result.height == 1
    assert result["pearson_ic"].to_list() == [None]
    assert result["spearman_ic"].to_list() == [None]


def test_spread_quantile_weights_handle_nulls_and_bucket_sizes() -> None:
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 4,
            "asset_id": ["a", "b", "c", "d"],
            "factor": [None, 1.0, 2.0, 3.0],
        }
    )

    weights = spread_quantile_weights(factor, quantiles=2)

    assert weights.select("asset_id", "weight").to_dicts() == [
        {"asset_id": "b", "weight": -1.0},
        {"asset_id": "c", "weight": 0.5},
        {"asset_id": "d", "weight": 0.5},
    ]


def test_factor_evaluation_drops_missing_price_keys_and_uses_matches() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                date
                for asset in ["a", "b", "c"]
                for date in ["2024-01-01", "2024-01-02", "2024-01-03"]
            ],
            "asset_id": [asset for asset in ["a", "b", "c"] for _ in range(3)],
            "price": [10.0, 11.0, 12.0, 20.0, 18.0, 16.0, 30.0, 33.0, 36.0],
        }
    )
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 4 + ["2024-01-04"],
            "asset_id": ["a", "b", "c", "d", "a"],
            "factor": [3.0, 1.0, 2.0, 4.0, 5.0],
        }
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )

    assert set(result.factor["asset_id"]) == {"a", "b", "c"}
    assert set(result.top_n_weights["asset_id"]) == {"a"}
    assert "d" not in set(result.spread_weights["asset_id"])
    assert result.ic.height == 1
    assert set(result.quantile_returns["quantile"]) == {"q1", "q2"}
    assert result.missing_price_keys.with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d")
    ).to_dicts() == [
        {"time": "2024-01-01", "asset_id": "d"},
        {"time": "2024-01-04", "asset_id": "a"},
    ]
    assert result.coverage.with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d")
    ).to_dicts() == [
        {
            "time": "2024-01-01",
            "factor_signal_asset_count": 4,
            "universe_asset_count": 3,
            "coverage_ratio": pytest.approx(4 / 3),
        },
        {
            "time": "2024-01-02",
            "factor_signal_asset_count": 0,
            "universe_asset_count": 3,
            "coverage_ratio": 0.0,
        },
        {
            "time": "2024-01-03",
            "factor_signal_asset_count": 0,
            "universe_asset_count": 3,
            "coverage_ratio": 0.0,
        },
    ]


def test_factor_evaluation_removes_null_and_nan_rows_before_alignment() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
            "asset_id": ["a", "a", "a", "b", "b", "b"],
            "price": [10.0, 11.0, 12.0, 20.0, float("nan"), 18.0],
        }
    )
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 4,
            "asset_id": ["a", "b", "c", "d"],
            "factor": [2.0, 1.0, None, float("nan")],
        }
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )

    assert result.factor.select("asset_id", "factor").to_dicts() == [
        {"asset_id": "a", "factor": 2.0},
        {"asset_id": "b", "factor": 1.0},
    ]
    assert result.missing_price_keys.is_empty()


def test_signal_diagnostics_build_only_requested_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = [date(2024, 1, 2), date(2024, 2, 1), date(2024, 3, 1)]
    prices = pl.DataFrame(
        {
            "time": [time for time in times for _ in range(4)],
            "asset_id": ["a", "b", "c", "d"] * len(times),
            "price": [
                10.0,
                20.0,
                30.0,
                40.0,
                11.0,
                18.0,
                33.0,
                38.0,
                12.0,
                17.0,
                36.0,
                37.0,
            ],
        }
    )
    signals = pl.DataFrame(
        {
            "time": [times[0]] * 4 + [times[1]] * 4,
            "asset_id": ["a", "b", "c", "d"] * 2,
            "signal": [4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    quantiles = materialize_signal_diagnostics(
        _scheduled_signal(signals),
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
        include_quantiles=True,
    )
    assert set(quantiles) == {"quantile_returns"}
    assert set(quantiles["quantile_returns"]["quantile"]) == {"q1", "q2"}

    spread = materialize_signal_diagnostics(
        _scheduled_signal(signals),
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
        include_spread=True,
    )
    assert set(spread) == {"spread_returns"}
    assert spread["spread_returns"].columns == ["time", "return"]

    original_batch = factor_module._run_sparse_compact_backtests
    batch_calls = 0

    def counted_batch(*args: object, **kwargs: object):
        nonlocal batch_calls
        batch_calls += 1
        return original_batch(*args, **kwargs)

    monkeypatch.setattr(
        factor_module,
        "_run_sparse_compact_backtests",
        counted_batch,
    )
    combined = materialize_signal_diagnostics(
        _scheduled_signal(signals),
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
        include_quantiles=True,
        include_lags=True,
    )
    assert set(combined) == {"quantile_returns", "lag_analysis", "lag_returns"}
    assert batch_calls == 1


def test_signal_diagnostics_skip_unrequested_family_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = [date(2024, 1, day) for day in range(1, 5)]
    prices = pl.DataFrame(
        {
            "time": times * 2,
            "asset_id": ["a"] * 4 + ["b"] * 4,
            "price": [10.0, 11.0, 12.0, 13.0, 10.0, 9.0, 8.0, 7.0],
        }
    )
    signals = pl.DataFrame(
        {
            "time": [times[0], times[0]],
            "asset_id": ["a", "b"],
            "signal": [2.0, 1.0],
        }
    )
    config = BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1)

    monkeypatch.setattr(
        factor_module,
        "_lag_outputs",
        lambda *args, **kwargs: pytest.fail("lag family must not be built"),
    )
    quantiles = materialize_signal_diagnostics(
        _scheduled_signal(signals),
        prices,
        config=config,
        include_quantiles=True,
    )
    assert set(quantiles) == {"quantile_returns"}

    monkeypatch.undo()
    monkeypatch.setattr(
        factor_module,
        "quantile_equal_weights",
        lambda *args, **kwargs: pytest.fail("quantile family must not be built"),
    )
    lags = materialize_signal_diagnostics(
        _scheduled_signal(signals),
        prices,
        config=config,
        include_lags=True,
    )
    assert set(lags) == {"lag_analysis", "lag_returns"}
