from __future__ import annotations

import matplotlib
import pandas as pd

from bagelquant_bt import BacktestConfig, run_factor_evaluation, run_weight_backtest
from bagelquant_bt.visualization import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_ic,
    plot_quantile_cumulative_returns,
    plot_rolling_ic,
    plot_turnover_and_costs,
)

matplotlib.use("Agg")


def test_visualization_functions_return_figures() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame(
        {"a": [100.0, 110.0, 121.0], "b": [100.0, 90.0, 99.0]},
        index=dates,
    )
    weights = pd.DataFrame({"a": [1.0, 0.0, 1.0], "b": [0.0, 1.0, 0.0]}, index=dates)
    factor = pd.DataFrame({"a": [2.0, 1.0, 2.0], "b": [1.0, 2.0, 1.0]}, index=dates)
    config = BacktestConfig(initial_capital=100_000, quantiles=2, top_n=1)

    backtest = run_weight_backtest(weights, prices, config=config)
    factor_result = run_factor_evaluation(factor, prices, config=config)

    assert plot_cumulative_returns(backtest)[0] is not None
    assert plot_drawdown(backtest)[0] is not None
    assert plot_turnover_and_costs(backtest)[0] is not None
    assert plot_ic(factor_result)[0] is not None
    assert plot_rolling_ic(factor_result)[0] is not None
    assert plot_quantile_cumulative_returns(factor_result)[0] is not None
