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


def test_non_price_weight_date_executes_on_next_price_date() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-03", "2024-01-04"],
            "asset_id": ["a", "a", "a"],
            "price": [10.0, 11.0, 12.0],
        }
    )
    weights = pl.DataFrame(
        {"time": ["2024-01-02"], "asset_id": ["a"], "weight": [1.0]}
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=10_000),
    )

    assert result.weights["time"].dt.strftime("%Y-%m-%d").to_list() == [
        "2024-01-03",
    ]


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (
            pl.DataFrame(
                {"time": ["2023-12-31"], "asset_id": ["a"], "weight": [1.0]}
            ),
            "outside covered price range",
        ),
        (
            pl.DataFrame(
                {"time": ["2024-01-04"], "asset_id": ["a"], "weight": [1.0]}
            ),
            "outside covered price range",
        ),
        (
            pl.DataFrame(
                {"time": ["2024-01-01"], "asset_id": ["b"], "weight": [1.0]}
            ),
            "assets missing from prices",
        ),
    ],
)
def test_weights_must_be_covered_by_prices(
    weights: pl.DataFrame,
    match: str,
) -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["a", "a"],
            "price": [10.0, 11.0],
        }
    )

    with pytest.raises(InputValidationError, match=match):
        run_weight_backtest(
            weights,
            prices,
            config=BacktestConfig(initial_capital=10_000),
        )
