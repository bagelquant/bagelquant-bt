from __future__ import annotations

import matplotlib
import pandas as pd

from bagelquant_bt import BacktestConfig, run_weight_backtest
from bagelquant_bt.visualization import plot_cumulative_returns, plot_drawdown

matplotlib.use("Agg")


dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
prices = pd.DataFrame(
    {"AAA": [100.0, 102.0, 101.0, 104.0], "BBB": [100.0, 99.0, 101.0, 102.0]},
    index=dates,
)
weights = pd.DataFrame(
    {"AAA": [0.7, 0.5, 0.3, 0.6], "BBB": [0.3, 0.5, 0.7, 0.4]},
    index=dates,
)

result = run_weight_backtest(
    weights,
    prices,
    config=BacktestConfig(initial_capital=1_000_000),
)

fig, _ = plot_cumulative_returns(result)
fig.savefig("cumulative_returns.png", dpi=150, bbox_inches="tight")

fig, _ = plot_drawdown(result)
fig.savefig("drawdown.png", dpi=150, bbox_inches="tight")

print("Saved cumulative_returns.png and drawdown.png")
