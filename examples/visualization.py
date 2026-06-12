from __future__ import annotations

import polars as pl

from bagelquant_bt import BacktestConfig, run_weight_backtest
from bagelquant_bt.visualization import plot_cumulative_returns, plot_drawdown

prices = pl.DataFrame(
    {
        "time": ["2024-01-02", "2024-01-03"] * 2,
        "asset_id": ["AAA", "AAA", "BBB", "BBB"],
        "price": [100.0, 102.0, 100.0, 99.0],
    }
)
weights = pl.DataFrame(
    {
        "time": ["2024-01-02", "2024-01-02"],
        "asset_id": ["AAA", "BBB"],
        "weight": [0.7, 0.3],
    }
)

result = run_weight_backtest(
    weights,
    prices,
    config=BacktestConfig(initial_capital=1_000_000),
)

plot_cumulative_returns(result).write_html("cumulative_returns.html")
plot_drawdown(result).write_html("drawdown.html")

print("Saved cumulative_returns.html and drawdown.html")
