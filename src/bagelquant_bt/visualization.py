"""Plotly visualization helpers for backtest and factor results."""

from __future__ import annotations

import math

import plotly.graph_objects as go

from .performance import rolling_performance
from .results import BacktestResult, FactorEvaluationResult
from .returns import drawdown


def plot_cumulative_returns(
    result: BacktestResult,
    *,
    title: str = "Cumulative Returns",
) -> go.Figure:
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
    fig.update_layout(title=title, yaxis_title="Return")
    return fig


def plot_drawdown(result: BacktestResult) -> go.Figure:
    gross = drawdown(result.returns, "gross_return")
    net = drawdown(result.returns, "net_return")
    fig = go.Figure()
    fig.add_scatter(
        x=gross["time"],
        y=gross["drawdown"],
        mode="lines",
        name=_label_with_mean("Gross drawdown", gross["drawdown"]),
    )
    fig.add_scatter(
        x=net["time"],
        y=net["drawdown"],
        mode="lines",
        name=_label_with_mean("Net drawdown", net["drawdown"]),
    )
    _add_average_lines(fig, [("Gross", gross["drawdown"]), ("Net", net["drawdown"])])
    fig.update_layout(title="Drawdown", yaxis_title="Drawdown")
    return fig


def plot_ic(result: FactorEvaluationResult) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=result.ic["time"],
        y=result.ic["pearson_ic"],
        mode="lines",
        name=_label_with_mean("Pearson IC", result.ic["pearson_ic"]),
    )
    fig.add_scatter(
        x=result.ic["time"],
        y=result.ic["spearman_ic"],
        mode="lines",
        name=_label_with_mean("Spearman IC", result.ic["spearman_ic"]),
    )
    _add_average_lines(
        fig,
        [
            ("Pearson", result.ic["pearson_ic"]),
            ("Spearman", result.ic["spearman_ic"]),
        ],
    )
    fig.add_hline(y=0.0, line_color="black", line_width=1)
    fig.update_layout(title="Information Coefficient", yaxis_title="IC")
    return fig


def plot_rolling_ic(result: FactorEvaluationResult, *, window: int = 20) -> go.Figure:
    data = result.ic.sort("time").with_columns(
        result.ic["pearson_ic"]
        .rolling_mean(window_size=window)
        .alias("rolling_pearson_ic"),
        result.ic["spearman_ic"]
        .rolling_mean(window_size=window)
        .alias("rolling_spearman_ic"),
    )
    fig = go.Figure()
    fig.add_scatter(
        x=data["time"],
        y=data["rolling_pearson_ic"],
        mode="lines",
        name=_label_with_mean("Rolling Pearson IC", data["rolling_pearson_ic"]),
    )
    fig.add_scatter(
        x=data["time"],
        y=data["rolling_spearman_ic"],
        mode="lines",
        name=_label_with_mean("Rolling Spearman IC", data["rolling_spearman_ic"]),
    )
    _add_average_lines(
        fig,
        [
            ("Pearson", data["rolling_pearson_ic"]),
            ("Spearman", data["rolling_spearman_ic"]),
        ],
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


def plot_rolling_sharpe(
    result: BacktestResult,
    *,
    annualization: int = 252,
) -> go.Figure:
    data = rolling_performance(result.returns, annualization=annualization)
    fig = go.Figure()
    for window in data["window"].unique().sort():
        frame = data.filter(data["window"] == window)
        label = _window_label(int(window), annualization)
        fig.add_scatter(
            x=frame["time"],
            y=frame["gross_sharpe"],
            mode="lines",
            name=_label_with_mean(f"Gross {label}", frame["gross_sharpe"]),
        )
        fig.add_scatter(
            x=frame["time"],
            y=frame["net_sharpe"],
            mode="lines",
            name=_label_with_mean(f"Net {label}", frame["net_sharpe"]),
        )
    _add_average_lines(
        fig,
        [("Gross", data["gross_sharpe"]), ("Net", data["net_sharpe"])],
    )
    fig.update_layout(title="Rolling Sharpe", yaxis_title="Sharpe")
    return fig


def plot_rolling_volatility(
    result: BacktestResult,
    *,
    annualization: int = 252,
) -> go.Figure:
    data = rolling_performance(result.returns, annualization=annualization)
    fig = go.Figure()
    for window in data["window"].unique().sort():
        frame = data.filter(data["window"] == window)
        label = _window_label(int(window), annualization)
        fig.add_scatter(
            x=frame["time"],
            y=frame["gross_volatility"],
            mode="lines",
            name=_label_with_mean(f"Gross {label}", frame["gross_volatility"]),
        )
        fig.add_scatter(
            x=frame["time"],
            y=frame["net_volatility"],
            mode="lines",
            name=_label_with_mean(f"Net {label}", frame["net_volatility"]),
        )
    _add_average_lines(
        fig,
        [("Gross", data["gross_volatility"]), ("Net", data["net_volatility"])],
    )
    fig.update_layout(title="Rolling Volatility", yaxis_title="Volatility")
    return fig


def plot_ic_distribution(result: FactorEvaluationResult) -> go.Figure:
    fig = go.Figure()
    pearson = result.ic["pearson_ic"].drop_nulls()
    spearman = result.ic["spearman_ic"].drop_nulls()
    fig.add_histogram(
        x=pearson,
        name=_label_with_mean_std("Pearson IC", pearson),
    )
    fig.add_histogram(
        x=spearman,
        name=_label_with_mean_std("Spearman IC", spearman),
    )
    _add_average_vlines(fig, [("Pearson", pearson), ("Spearman", spearman)])
    fig.update_layout(
        title="IC Distribution",
        xaxis_title="IC",
        yaxis_title="Count",
        barmode="overlay",
    )
    return fig


def plot_lag_cumulative_return(result: FactorEvaluationResult) -> list[go.Figure]:
    """Plot lagged cumulative return time series by portfolio and return type."""

    return [
        _plot_lag_return_path(
            result,
            portfolio="top_n",
            return_type="gross",
            title="TOP N Gross Lag Cumulative Returns",
        ),
        _plot_lag_return_path(
            result,
            portfolio="top_n",
            return_type="net",
            title="TOP N Net Lag Cumulative Returns",
        ),
        _plot_lag_return_path(
            result,
            portfolio="long_short",
            return_type="gross",
            title="Long-Short Gross Lag Cumulative Returns",
        ),
        _plot_lag_return_path(
            result,
            portfolio="long_short",
            return_type="net",
            title="Long-Short Net Lag Cumulative Returns",
        ),
    ]


def plot_lag_sharpe(result: FactorEvaluationResult) -> go.Figure:
    fig = go.Figure()
    for portfolio in result.lag_analysis["portfolio"].unique().sort():
        data = result.lag_analysis.filter(result.lag_analysis["portfolio"] == portfolio)
        fig.add_scatter(
            x=data["lag"],
            y=data["gross_sharpe"],
            mode="lines+markers",
            name=f"{portfolio} gross",
        )
        fig.add_scatter(
            x=data["lag"],
            y=data["net_sharpe"],
            mode="lines+markers",
            name=f"{portfolio} net",
        )
    fig.update_layout(
        title="Lag Analysis Sharpe",
        xaxis_title="Lag",
        yaxis_title="Sharpe",
    )
    return fig


def plot_ic_decay(result: FactorEvaluationResult) -> go.Figure:
    fig = go.Figure()
    for method in result.ic_decay["method"].unique().sort():
        data = result.ic_decay.filter(result.ic_decay["method"] == method)
        fig.add_scatter(
            x=data["lag"],
            y=data["ic_mean"],
            mode="lines+markers",
            name=f"{method} IC",
        )
    fig.update_layout(
        title="IC Decay",
        xaxis_title="Lag",
        yaxis_title="Mean IC",
    )
    return fig


def plot_ic_decay_heatmap(result: FactorEvaluationResult) -> go.Figure:
    """Compatibility alias for the IC decay line chart."""

    return plot_ic_decay(result)


def _plot_lag_return_path(
    result: FactorEvaluationResult,
    *,
    portfolio: str,
    return_type: str,
    title: str,
) -> go.Figure:
    value_column = f"{return_type}_cumulative_return"
    sharpe_column = f"{return_type}_sharpe"
    data = result.lag_returns.filter(result.lag_returns["portfolio"] == portfolio)
    fig = go.Figure()
    for lag in data["lag"].unique().sort():
        frame = data.filter(data["lag"] == lag)
        sharpe = frame[sharpe_column].drop_nulls()
        sharpe_value = float(sharpe[0]) if len(sharpe) else math.nan
        fig.add_scatter(
            x=frame["time"],
            y=frame[value_column],
            mode="lines",
            name=f"Lag {int(lag)} (Sharpe {sharpe_value:.4f})",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Return",
    )
    return fig


def _add_average_lines(fig: go.Figure, series: list[tuple[str, object]]) -> None:
    for name, values in series:
        finite_values = _finite_values(values)
        if not finite_values:
            continue
        value = sum(finite_values) / len(finite_values)
        fig.add_hline(
            y=value,
            line_dash="dot",
            name=f"{name} avg",
            showlegend=False,
        )


def _add_average_vlines(fig: go.Figure, series: list[tuple[str, object]]) -> None:
    for name, values in series:
        finite_values = _finite_values(values)
        if not finite_values:
            continue
        value = sum(finite_values) / len(finite_values)
        fig.add_vline(
            x=value,
            line_dash="dot",
            name=f"{name} avg",
            showlegend=False,
        )


def _label_with_mean(label: str, values: object) -> str:
    finite_values = _finite_values(values)
    if not finite_values:
        return label
    mean = sum(finite_values) / len(finite_values)
    return f"{label} (avg {mean:.4f})"


def _label_with_mean_std(label: str, values: object) -> str:
    finite_values = _finite_values(values)
    if not finite_values:
        return label
    mean = sum(finite_values) / len(finite_values)
    std = math.nan
    if len(finite_values) > 1:
        variance = sum((value - mean) ** 2 for value in finite_values) / (
            len(finite_values) - 1
        )
        std = math.sqrt(variance)
    return f"{label} (avg {mean:.4f}, std {std:.4f})"


def _finite_values(values: object) -> list[float]:
    return [
        float(value)
        for value in values.drop_nulls()
        if math.isfinite(float(value))
    ]


def _window_label(window: int, annualization: int) -> str:
    if window == annualization:
        return "12M"
    if window == max(1, annualization // 2):
        return "6M"
    return f"{window}"
