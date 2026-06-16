from __future__ import annotations

import polars as pl

from bagelquant_bt import BacktestConfig, run_weight_backtest


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
