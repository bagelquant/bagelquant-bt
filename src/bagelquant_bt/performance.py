"""Pure rolling volatility and Sharpe over saved Gross/Net returns."""

from __future__ import annotations

import math

import polars as pl

from .exceptions import InputValidationError
from .statistics import return_metrics


def rolling_performance(
    returns: pl.DataFrame,
    *,
    annualization: int,
    windows: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    """Compute finite rolling Gross/Net metrics with unavailable reasons."""

    return_metrics([], annualization)
    if windows is None:
        windows = (max(1, annualization // 2), annualization)
    if any(
        isinstance(window, bool) or not isinstance(window, int) or window < 1
        for window in windows
    ):
        raise InputValidationError("rolling windows must be positive integers")
    data = returns.sort("time").with_columns(
        pl.when(pl.col(column).is_finite())
        .then(pl.col(column))
        .otherwise(None)
        .alias(column)
        for column in ("gross_return", "net_return")
    )
    frames: list[pl.DataFrame] = []
    for window in windows:
        expressions = [pl.col("time"), pl.lit(window).alias("window")]
        for prefix in ("gross", "net"):
            column = f"{prefix}_return"
            volatility = pl.col(column).rolling_std(window_size=window) * math.sqrt(
                annualization
            )
            expressions.extend(
                (
                    pl.when(volatility.is_finite())
                    .then(volatility)
                    .otherwise(None)
                    .alias(f"{prefix}_volatility"),
                    _rolling_sharpe_expr(column, window, annualization).alias(
                        f"{prefix}_sharpe"
                    ),
                    _rolling_reason(column, window, annualization, ratio=False).alias(
                        f"{prefix}_volatility_reason"
                    ),
                    _rolling_reason(column, window, annualization, ratio=True).alias(
                        f"{prefix}_sharpe_reason"
                    ),
                )
            )
        frames.append(data.select(expressions))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).sort(["window", "time"])


def _rolling_sharpe_expr(
    column: str,
    window: int,
    annualization: int,
) -> pl.Expr:
    rolling_std = pl.col(column).rolling_std(window_size=window)
    ratio = (
        pl.col(column).rolling_mean(window_size=window)
        / rolling_std
        * math.sqrt(annualization)
    )
    return pl.when((rolling_std != 0) & ratio.is_finite()).then(ratio).otherwise(None)


def _rolling_reason(
    column: str, window: int, annualization: int, *, ratio: bool
) -> pl.Expr:
    count = (
        pl.col(column)
        .is_not_null()
        .cast(pl.Int64)
        .rolling_sum(window_size=window, min_samples=1)
    )
    deviation = pl.col(column).rolling_std(window_size=window)
    reason = pl.when(count < window).then(
        pl.lit("insufficient finite observations for rolling window")
    )
    if window < 2:
        return reason.otherwise(pl.lit("at least two return observations required"))
    reason = reason.when(~deviation.is_finite()).then(
        pl.lit("sample standard deviation is non-finite")
    )
    if ratio:
        value = (
            pl.col(column).rolling_mean(window_size=window)
            / deviation
            * math.sqrt(annualization)
        )
        reason = reason.when(deviation == 0).then(pl.lit("sample variance is zero"))
        message = "Sharpe ratio is non-finite"
    else:
        value = deviation * math.sqrt(annualization)
        message = "annualized volatility is non-finite"
    return (
        reason.when(~value.is_finite())
        .then(pl.lit(message))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
