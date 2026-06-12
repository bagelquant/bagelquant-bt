"""Performance summary helpers."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .results import PerformanceSummary, TransactionCostBreakdown
from .returns import drawdown


def summarize_performance(
    *,
    returns: pl.DataFrame,
    turnover: pl.DataFrame,
    costs: TransactionCostBreakdown,
    initial_capital: float,
    annualization: int,
) -> PerformanceSummary:
    """Summarize net performance while retaining gross/net final values."""

    frame = returns.sort("time")
    net = np.array(frame["net_return"].fill_null(0.0), dtype=float)
    gross = np.array(frame["gross_return"].fill_null(0.0), dtype=float)
    periods = len(net)
    final_net_value = initial_capital * float(np.prod(1.0 + net))
    final_gross_value = initial_capital * float(np.prod(1.0 + gross))
    total_return = final_net_value / initial_capital - 1.0
    annualized_return = (
        (1.0 + total_return) ** (annualization / periods) - 1.0
        if periods > 0
        else math.nan
    )
    net_std = float(np.std(net, ddof=1)) if periods > 1 else math.nan
    net_mean = float(np.mean(net)) if periods else math.nan
    annualized_volatility = net_std * math.sqrt(annualization)
    sharpe = (
        net_mean / net_std * math.sqrt(annualization)
        if net_std != 0 and not math.isnan(net_std)
        else math.nan
    )
    dd = drawdown(frame, "net_return")
    max_drawdown = float(dd["drawdown"].min()) if periods else math.nan
    hit_rate = float(np.mean(net > 0)) if periods else math.nan

    return PerformanceSummary(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        annualized_volatility=float(annualized_volatility),
        sharpe=float(sharpe),
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        average_turnover=(
            float(turnover["turnover"].mean()) if turnover.height else math.nan
        ),
        total_transaction_cost=(
            float(costs.data["total_fee"].sum()) if costs.data.height else 0.0
        ),
        final_gross_value=float(final_gross_value),
        final_net_value=float(final_net_value),
    )
