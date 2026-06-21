from __future__ import annotations

import polars as pl
import pytest

from bagelquant_bt import BacktestConfig, TransactionCostConfig, run_weight_backtest
from bagelquant_bt.exceptions import InputValidationError


def test_weight_backtest_returns_polars_result_frames() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["a", "a", "a"],
            "price": [10.0, 11.0, 12.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["a", "a"],
            "weight": [1.0, 1.0],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=100_000, annualization=252),
    )

    assert result.returns.columns == ["time", "gross_return", "net_return"]
    assert result.value.columns == [
        "time",
        "gross_value",
        "net_value",
        "gross_return_cumulative",
        "net_return_cumulative",
    ]
    assert result.transaction_costs.data["total_fee"].sum() > 0
    assert result.summary.gross_sharpe != result.summary.net_sharpe
    assert result.summary.gross_max_drawdown >= result.summary.net_max_drawdown
    assert result.summary.sharpe == result.summary.net_sharpe
    assert result.summary.max_drawdown == result.summary.net_max_drawdown
    assert result.performance.columns == ["metric", "gross", "net"]
    assert "sharpe" in result.performance["metric"].to_list()


def test_low_frequency_weights_hold_until_next_rebalance() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "asset_id": ["a", "a", "a", "a", "a"],
            "price": [100.0, 110.0, 121.0, 133.1, 146.41],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-03"],
            "asset_id": ["a", "a"],
            "weight": [1.0, 0.5],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(
            initial_capital=10_000,
            annualization=252,
            transaction_cost=TransactionCostConfig(rate=0.001, min_fee=0.0),
        ),
    )

    assert result.weights["weight"].to_list() == [1.0, 1.0, 0.5, 0.5]
    assert result.returns["gross_return"].round(3).to_list() == [0.1, 0.1, 0.05, 0.05]
    assert result.turnover["turnover"].to_list() == [1.0, 0.0, 0.5, 0.0]
    assert result.transaction_costs.data["total_fee"].to_list()[1] == 0.0
    assert result.transaction_costs.data["total_fee"].to_list()[3] == 0.0


def test_transaction_cost_min_fee_is_applied_per_traded_asset() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"] * 2,
            "asset_id": ["a", "a", "b", "b"],
            "price": [10.0, 10.0, 20.0, 20.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "weight": [0.5, 0.5],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(
            initial_capital=1_000,
            transaction_cost=TransactionCostConfig(rate=0.001, min_fee=5.0),
        ),
    )

    cost = result.transaction_costs.data.to_dicts()[0]
    assert cost["traded_asset_count"] == 2
    assert cost["traded_notional"] == pytest.approx(1_000.0)
    assert cost["raw_fee"] == pytest.approx(1.0)
    assert cost["total_fee"] == pytest.approx(10.0)
    assert cost["min_fee_adjustment"] == pytest.approx(9.0)


def test_weight_backtest_raises_when_transaction_costs_exhaust_capital() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"] * 3,
            "asset_id": ["a", "a", "b", "b", "c", "c"],
            "price": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b", "c"],
            "weight": [1 / 3, 1 / 3, 1 / 3],
        }
    )

    with pytest.raises(
        InputValidationError,
        match=(
            r"net portfolio value became non-positive.*"
            "Increase initial_capital or reduce traded universe/turnover"
        ),
    ):
        run_weight_backtest(
            weights,
            prices,
            config=BacktestConfig(
                initial_capital=10,
                transaction_cost=TransactionCostConfig(rate=0.001, min_fee=5.0),
            ),
        )


def test_non_price_weight_date_is_dropped() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-03", "2024-01-04"],
            "asset_id": ["a", "a", "a"],
            "price": [10.0, 11.0, 12.0],
        }
    )
    weights = pl.DataFrame({"time": ["2024-01-02"], "asset_id": ["a"], "weight": [1.0]})

    with pytest.raises(InputValidationError, match="at least two overlapping"):
        run_weight_backtest(
            weights,
            prices,
            config=BacktestConfig(initial_capital=10_000),
        )


def test_weight_backtest_drops_missing_price_keys_and_trades_matches() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
            "asset_id": ["a", "a", "a", "b", "b", "b"],
            "price": [10.0, 11.0, 12.0, 20.0, 18.0, 16.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-04",
            ],
            "asset_id": ["a", "b", "c", "a"],
            "weight": [0.5, 0.5, 0.5, 1.0],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=10_000),
    )

    assert set(result.weights["asset_id"]) == {"a", "b"}
    assert result.missing_price_keys.with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d")
    ).to_dicts() == [
        {"time": "2024-01-01", "asset_id": "c"},
        {"time": "2024-01-04", "asset_id": "a"},
    ]
    assert result.returns["gross_return"].round(4).to_list() == [
        0.0,
        pytest.approx(-0.0101),
    ]
    assert result.coverage.with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d")
    ).to_dicts() == [
        {
            "time": "2024-01-01",
            "weight_asset_count": 3,
            "universe_asset_count": 2,
            "coverage_ratio": 1.5,
        },
        {
            "time": "2024-01-02",
            "weight_asset_count": 0,
            "universe_asset_count": 2,
            "coverage_ratio": 0.0,
        },
        {
            "time": "2024-01-03",
            "weight_asset_count": 0,
            "universe_asset_count": 2,
            "coverage_ratio": 0.0,
        },
    ]


def test_weight_backtest_removes_null_and_nan_rows_before_alignment() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["a", "a", "a"],
            "price": [10.0, float("nan"), 12.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["a", "a", "a"],
            "weight": [1.0, None, float("nan")],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=10_000),
    )

    assert result.weights.with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d")
    ).to_dicts() == [{"time": "2024-01-01", "asset_id": "a", "weight": 1.0}]
    assert result.missing_price_keys.is_empty()
