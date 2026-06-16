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
) -> tuple[PerformanceSummary, pl.DataFrame]:
    """Summarize net performance while retaining gross/net final values."""

    frame = returns.sort("time")
    net = np.array(frame["net_return"].fill_null(0.0), dtype=float)
    periods = len(net)

    gross_metrics = _return_metrics(
        frame,
        "gross_return",
        initial_capital=initial_capital,
        annualization=annualization,
    )
    net_metrics = _return_metrics(
        frame,
        "net_return",
        initial_capital=initial_capital,
        annualization=annualization,
    )
    hit_rate = float(np.mean(net > 0)) if periods else math.nan
    average_turnover = (
        float(turnover["turnover"].mean()) if turnover.height else math.nan  # type: ignore
    )
    total_transaction_cost = (
        float(costs.data["total_fee"].sum()) if costs.data.height else 0.0
    )

    summary = PerformanceSummary(
        total_return=net_metrics["total_return"],
        annualized_return=net_metrics["annualized_return"],
        annualized_volatility=net_metrics["annualized_volatility"],
        sharpe=net_metrics["sharpe"],
        max_drawdown=net_metrics["max_drawdown"],
        gross_total_return=gross_metrics["total_return"],
        net_total_return=net_metrics["total_return"],
        gross_annualized_return=gross_metrics["annualized_return"],
        net_annualized_return=net_metrics["annualized_return"],
        gross_annualized_volatility=gross_metrics["annualized_volatility"],
        net_annualized_volatility=net_metrics["annualized_volatility"],
        gross_sharpe=gross_metrics["sharpe"],
        net_sharpe=net_metrics["sharpe"],
        gross_max_drawdown=gross_metrics["max_drawdown"],
        net_max_drawdown=net_metrics["max_drawdown"],
        hit_rate=hit_rate,
        average_turnover=average_turnover,
        total_transaction_cost=total_transaction_cost,
        final_gross_value=gross_metrics["final_value"],
        final_net_value=net_metrics["final_value"],
    )
    return summary, performance_matrix(summary)


def performance_matrix(summary: PerformanceSummary) -> pl.DataFrame:
    """Return a gross/net metric matrix suitable for display or export."""

    return pl.DataFrame(
        [
            {
                "metric": "total_return",
                "gross": summary.gross_total_return,
                "net": summary.net_total_return,
            },
            {
                "metric": "annualized_return",
                "gross": summary.gross_annualized_return,
                "net": summary.net_annualized_return,
            },
            {
                "metric": "annualized_volatility",
                "gross": summary.gross_annualized_volatility,
                "net": summary.net_annualized_volatility,
            },
            {
                "metric": "sharpe",
                "gross": summary.gross_sharpe,
                "net": summary.net_sharpe,
            },
            {
                "metric": "max_drawdown",
                "gross": summary.gross_max_drawdown,
                "net": summary.net_max_drawdown,
            },
            {"metric": "hit_rate", "gross": None, "net": summary.hit_rate},
            {
                "metric": "final_value",
                "gross": summary.final_gross_value,
                "net": summary.final_net_value,
            },
            {
                "metric": "average_turnover",
                "gross": summary.average_turnover,
                "net": summary.average_turnover,
            },
            {
                "metric": "total_transaction_cost",
                "gross": None,
                "net": summary.total_transaction_cost,
            },
        ],
        schema={
            "metric": pl.String,
            "gross": pl.Float64,
            "net": pl.Float64,
        },
    )


def rolling_performance(
    returns: pl.DataFrame,
    *,
    annualization: int,
    windows: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    """Compute rolling gross/net volatility and Sharpe."""

    if windows is None:
        windows = (max(1, annualization // 2), annualization)
    data = returns.sort("time")
    frames: list[pl.DataFrame] = []
    for window in windows:
        frames.append(
            data.select(
                pl.col("time"),
                pl.lit(window).alias("window"),
                (
                    pl.col("gross_return").rolling_std(window_size=window)
                    * math.sqrt(annualization)
                ).alias("gross_volatility"),
                (
                    pl.col("net_return").rolling_std(window_size=window)
                    * math.sqrt(annualization)
                ).alias("net_volatility"),
                _rolling_sharpe_expr("gross_return", window, annualization).alias(
                    "gross_sharpe"
                ),
                _rolling_sharpe_expr("net_return", window, annualization).alias(
                    "net_sharpe"
                ),
            )
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).sort(["window", "time"])


def _return_metrics(
    frame: pl.DataFrame,
    column: str,
    *,
    initial_capital: float,
    annualization: int,
) -> dict[str, float]:
    values = np.array(frame[column].fill_null(0.0), dtype=float)
    periods = len(values)
    final_value = initial_capital * float(np.prod(1.0 + values))
    total_return = final_value / initial_capital - 1.0
    annualized_return = (
        (1.0 + total_return) ** (annualization / periods) - 1.0
        if periods > 0
        else math.nan
    )
    std = float(np.std(values, ddof=1)) if periods > 1 else math.nan
    mean = float(np.mean(values)) if periods else math.nan
    annualized_volatility = std * math.sqrt(annualization)
    sharpe = (
        mean / std * math.sqrt(annualization)
        if std != 0 and not math.isnan(std)
        else math.nan
    )
    dd = drawdown(frame, column)
    max_drawdown = float(dd["drawdown"].min()) if periods else math.nan  # type: ignore
    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "final_value": float(final_value),
    }


def _rolling_sharpe_expr(
    column: str,
    window: int,
    annualization: int,
) -> pl.Expr:
    rolling_std = pl.col(column).rolling_std(window_size=window)
    return (
        pl.when(rolling_std != 0)
        .then(
            pl.col(column).rolling_mean(window_size=window)
            / rolling_std
            * math.sqrt(annualization)
        )
        .otherwise(None)
    )
