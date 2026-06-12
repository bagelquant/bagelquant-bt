"""Plotly visualization helpers for backtest and factor results."""

from __future__ import annotations

import plotly.graph_objects as go

from .results import BacktestResult, FactorEvaluationResult
from .returns import drawdown


def plot_cumulative_returns(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=result.value["time"],
        y=result.value["gross_return_cumulative"],
        mode="lines",
        name="Gross",
    )
    fig.add_scatter(
        x=result.value["time"],
        y=result.value["net_return_cumulative"],
        mode="lines",
        name="Net",
    )
    fig.update_layout(title="Cumulative Returns", yaxis_title="Return")
    return fig


def plot_drawdown(result: BacktestResult) -> go.Figure:
    data = drawdown(result.returns, "net_return")
    fig = go.Figure()
    fig.add_scatter(
        x=data["time"], y=data["drawdown"], mode="lines", name="Net drawdown"
    )
    fig.update_layout(title="Drawdown", yaxis_title="Drawdown")
    return fig


def plot_ic(result: FactorEvaluationResult) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=result.ic["time"], y=result.ic["ic"], mode="lines", name="IC")
    fig.add_hline(y=0.0, line_color="black", line_width=1)
    fig.update_layout(title="Information Coefficient", yaxis_title="IC")
    return fig


def plot_rolling_ic(result: FactorEvaluationResult, *, window: int = 20) -> go.Figure:
    data = result.ic.sort("time").with_columns(
        result.ic["ic"]
        .rolling_mean(window_size=window, min_samples=1)
        .alias("rolling_ic")
    )
    fig = go.Figure()
    fig.add_scatter(
        x=data["time"], y=data["rolling_ic"], mode="lines", name="Rolling IC"
    )
    fig.add_hline(y=0.0, line_color="black", line_width=1)
    fig.update_layout(title="Rolling IC", yaxis_title="IC")
    return fig


def plot_quantile_cumulative_returns(result: FactorEvaluationResult) -> go.Figure:
    fig = go.Figure()
    for quantile in result.quantile_returns["quantile"].unique().sort():
        data = result.quantile_returns.filter(
            result.quantile_returns["quantile"] == quantile
        )
        fig.add_scatter(
            x=data["time"],
            y=data["cumulative_return"],
            mode="lines",
            name=str(quantile),
        )
    fig.update_layout(title="Quantile Cumulative Returns", yaxis_title="Return")
    return fig


def plot_turnover_and_costs(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=result.turnover["time"],
        y=result.turnover["turnover"],
        mode="lines",
        name="Turnover",
        yaxis="y1",
    )
    fig.add_scatter(
        x=result.transaction_costs.data["time"],
        y=result.transaction_costs.data["total_fee"],
        mode="lines",
        name="Transaction costs",
        yaxis="y2",
    )
    fig.update_layout(
        title="Turnover and Transaction Costs",
        yaxis={"title": "Turnover"},
        yaxis2={"title": "Fee", "overlaying": "y", "side": "right"},
    )
    return fig
