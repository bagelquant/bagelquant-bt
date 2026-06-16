from __future__ import annotations

import plotly.graph_objects as go
import polars as pl

from bagelquant_bt import BacktestConfig, run_factor_evaluation, run_weight_backtest
from bagelquant_bt.visualization import (
    plot_cumulative_returns,
    plot_ic,
    plot_ic_decay,
    plot_ic_decay_heatmap,
    plot_ic_distribution,
    plot_lag_cumulative_return,
    plot_lag_sharpe,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)


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

    fig = plot_cumulative_returns(result)
    assert isinstance(fig, go.Figure)
    assert fig.layout.shapes == ()


def test_new_visualization_helpers_return_plotly_figures() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "asset_id": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
            "price": [1.0, 1.1, 1.2, 1.3, 1.0, 0.9, 0.8, 0.7, 1.0, 1.0, 1.1, 1.2],
        }
    )
    weights = prices.select("time", "asset_id").with_columns(
        pl.lit(1 / 3).alias("weight")
    )
    backtest = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=1_000, annualization=4),
    )
    factor = prices.select("time", "asset_id").with_columns(
        pl.when(pl.col("asset_id") == "a")
        .then(3.0)
        .when(pl.col("asset_id") == "c")
        .then(2.0)
        .otherwise(1.0)
        .alias("factor")
    )
    factor_result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=1_000, annualization=4, quantiles=3),
    )

    lag_cumulative = plot_lag_cumulative_return(factor_result)
    ic_distribution = plot_ic_distribution(factor_result)
    ic_decay = plot_ic_decay(factor_result)
    ic_decay_alias = plot_ic_decay_heatmap(factor_result)
    ic = plot_ic(factor_result)
    figures = [
        plot_rolling_sharpe(backtest, annualization=4),
        plot_rolling_volatility(backtest, annualization=4),
        ic_distribution,
        *lag_cumulative,
        plot_lag_sharpe(factor_result),
        ic_decay,
        ic_decay_alias,
        ic,
    ]

    assert all(isinstance(fig, go.Figure) for fig in figures)
    assert len(lag_cumulative) == 4
    assert all(fig.layout.shapes == () for fig in lag_cumulative)
    assert all("Sharpe" in trace.name for fig in lag_cumulative for trace in fig.data)
    assert all("avg" in trace.name for trace in ic.data)
    assert ic.layout.annotations == ()
    assert len(ic_distribution.layout.shapes) == 2
    assert all(
        "avg" in trace.name and "std" in trace.name
        for trace in ic_distribution.data
    )
    assert all(trace.type == "scatter" for trace in ic_decay.data)
    assert all(trace.type != "heatmap" for trace in ic_decay_alias.data)
    assert ic_decay.layout.shapes == ()
