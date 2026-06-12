"""Return-series utilities for long-form Polars panels."""

from __future__ import annotations

import polars as pl

from .inputs import ASSET_ID, TIME


def asset_close_to_close_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute close-to-close returns per asset."""

    return (
        prices.sort([ASSET_ID, TIME])
        .with_columns(
            (pl.col("price").shift(-1).over(ASSET_ID) / pl.col("price") - 1.0).alias(
                "forward_return"
            )
        )
        .filter(pl.col("forward_return").is_not_null())
        .select(TIME, ASSET_ID, "forward_return")
        .sort([TIME, ASSET_ID])
    )


def align_weights_to_forward_returns(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    returns = asset_close_to_close_returns(prices)
    executable = weights.join(
        returns.select(TIME, ASSET_ID),
        on=[TIME, ASSET_ID],
        how="inner",
    ).sort([TIME, ASSET_ID])
    return executable, returns


def portfolio_returns(
    weights: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    return (
        weights.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
        .with_columns(
            (
                pl.col("weight").fill_null(0.0)
                * pl.col("forward_return").fill_null(0.0)
            ).alias("weighted_return")
        )
        .group_by(TIME)
        .agg(pl.col("weighted_return").sum().alias("gross_return"))
        .sort(TIME)
    )


def cumulative_returns(returns: pl.DataFrame, column: str) -> pl.DataFrame:
    return returns.select(
        TIME,
        ((1.0 + pl.col(column).fill_null(0.0)).cum_prod() - 1.0).alias(
            f"{column}_cumulative"
        ),
    )


def value_path(
    returns: pl.DataFrame,
    column: str,
    *,
    initial_capital: float,
    output_column: str,
) -> pl.DataFrame:
    return returns.select(
        TIME,
        (initial_capital * (1.0 + pl.col(column).fill_null(0.0)).cum_prod()).alias(
            output_column
        ),
    )


def drawdown(returns: pl.DataFrame, column: str) -> pl.DataFrame:
    wealth = (1.0 + pl.col(column).fill_null(0.0)).cum_prod()
    return returns.select(TIME, (wealth / wealth.cum_max() - 1.0).alias("drawdown"))
