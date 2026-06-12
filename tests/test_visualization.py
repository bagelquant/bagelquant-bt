from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

from bagelquant_bt import BacktestConfig, run_weight_backtest
from bagelquant_bt.visualization import plot_cumulative_returns


def test_visualization_returns_plotly_figure() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["a", "a"],
            "price": [1.0, 2.0],
        }
    )
    weights = pl.DataFrame({"time": ["2024-01-01"], "asset_id": ["a"], "weight": [1.0]})
    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=1_000),
    )

    assert isinstance(plot_cumulative_returns(result), go.Figure)
