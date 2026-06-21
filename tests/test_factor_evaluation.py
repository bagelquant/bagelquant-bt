from __future__ import annotations

import polars as pl
import pytest

from bagelquant_bt import BacktestConfig, run_factor_evaluation
from bagelquant_bt.factor import (
    factor_quantile_returns,
    information_coefficients,
    spread_quantile_weights,
)


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
