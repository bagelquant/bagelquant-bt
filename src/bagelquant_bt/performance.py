"""Performance summary helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .results import PerformanceSummary, TransactionCostBreakdown
from .returns import drawdown


def summarize_performance(
    *,
    gross_returns: pd.Series,
    net_returns: pd.Series,
    turnover: pd.Series,
    costs: TransactionCostBreakdown,
    initial_capital: float,
    annualization: int,
) -> PerformanceSummary:
    """Summarize net performance while retaining gross/net final values."""

    net = net_returns.fillna(0.0)
    gross = gross_returns.fillna(0.0)
    final_net_value = initial_capital * float((1.0 + net).prod())
    final_gross_value = initial_capital * float((1.0 + gross).prod())
    total_return = final_net_value / initial_capital - 1.0
    periods = len(net)
    annualized_return = (
        (1.0 + total_return) ** (annualization / periods) - 1.0
        if periods > 0
        else np.nan
    )
    annualized_volatility = float(net.std(ddof=1) * np.sqrt(annualization))
    sharpe = (
        float(net.mean() / net.std(ddof=1) * np.sqrt(annualization))
        if net.std(ddof=1) != 0 and not np.isnan(net.std(ddof=1))
        else np.nan
    )
    max_drawdown = float(drawdown(net).min()) if periods else np.nan
    hit_rate = float(net.gt(0).mean()) if periods else np.nan

    return PerformanceSummary(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        average_turnover=float(turnover.mean()) if len(turnover) else np.nan,
        total_transaction_cost=float(costs.total_fee.sum()),
        final_gross_value=float(final_gross_value),
        final_net_value=float(final_net_value),
    )
