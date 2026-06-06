"""Matplotlib visualization helpers for backtest and factor results."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .results import BacktestResult, FactorEvaluationResult


def plot_cumulative_returns(result: BacktestResult):
    fig, ax = plt.subplots()
    result.gross_cumulative_returns.plot(ax=ax, label="Gross")
    result.net_cumulative_returns.plot(ax=ax, label="Net")
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Return")
    ax.legend()
    return fig, ax


def plot_drawdown(result: BacktestResult):
    from .returns import drawdown

    fig, ax = plt.subplots()
    drawdown(result.net_returns).plot(ax=ax, label="Net drawdown")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.legend()
    return fig, ax


def plot_ic(result: FactorEvaluationResult):
    fig, ax = plt.subplots()
    result.ic.plot(ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Information Coefficient")
    ax.set_ylabel("IC")
    return fig, ax


def plot_rolling_ic(result: FactorEvaluationResult, *, window: int = 20):
    fig, ax = plt.subplots()
    result.ic.rolling(window=window, min_periods=1).mean().plot(ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Rolling IC")
    ax.set_ylabel("IC")
    return fig, ax


def plot_quantile_cumulative_returns(result: FactorEvaluationResult):
    fig, ax = plt.subplots()
    result.quantile_cumulative_returns.plot(ax=ax)
    ax.set_title("Quantile Cumulative Returns")
    ax.set_ylabel("Return")
    ax.legend()
    return fig, ax


def plot_turnover_and_costs(result: BacktestResult):
    fig, axes = plt.subplots(2, 1, sharex=True)
    result.turnover.plot(ax=axes[0])
    result.transaction_costs.total_fee.plot(ax=axes[1])
    axes[0].set_title("Turnover")
    axes[0].set_ylabel("Turnover")
    axes[1].set_title("Transaction Costs")
    axes[1].set_ylabel("Fee")
    return fig, axes
