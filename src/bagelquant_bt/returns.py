"""Return-series utilities for long-form Polars panels."""

from __future__ import annotations

import warnings

import polars as pl

from .exceptions import InputValidationError
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
    executable = _expand_portfolio_weights(weights, prices, returns)
    return executable, returns


def align_signal_to_forward_returns(
    signal: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    value_column: str,
    label: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep signal snapshots with exact price keys."""

    returns = asset_close_to_close_returns(prices)
    aligned = _filter_snapshots_to_price_keys(
        signal,
        prices,
        value_columns=(value_column,),
        label=label,
    )
    return aligned, returns


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


def _expand_portfolio_weights(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    aligned = _filter_snapshots_to_price_keys(
        weights,
        prices,
        value_columns=("weight",),
        label="weights",
    )
    if aligned.is_empty():
        return aligned

    first_execution_time = aligned.get_column(TIME).min()
    execution_keys = (
        prices.select(TIME, ASSET_ID)
        .join(forward_returns.select(TIME, ASSET_ID), on=[TIME, ASSET_ID], how="inner")
        .filter(pl.col(TIME) >= first_execution_time)
        .unique()
        .sort([ASSET_ID, TIME])
    )
    snapshot_times = aligned.select(TIME).unique()
    snapshot_assets = aligned.select(ASSET_ID).unique()
    dense_snapshots = (
        snapshot_times.join(snapshot_assets, how="cross")
        .join(aligned, on=[TIME, ASSET_ID], how="left")
        .with_columns(pl.col("weight").fill_null(0.0))
        .sort([ASSET_ID, TIME])
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        expanded = execution_keys.join_asof(
            dense_snapshots,
            on=TIME,
            by=ASSET_ID,
            strategy="backward",
        )
    return (
        expanded
        .drop_nulls("weight")
        .select(TIME, ASSET_ID, "weight")
        .sort([TIME, ASSET_ID])
    )


def _filter_snapshots_to_price_keys(
    frame: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    value_columns: tuple[str, ...],
    label: str,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame.select(TIME, ASSET_ID, *value_columns)
    if prices.is_empty():
        raise InputValidationError(f"{label} has no covered price range")

    aligned = frame.join(
        prices.select(TIME, ASSET_ID),
        on=[TIME, ASSET_ID],
        how="inner",
    )
    return aligned.select(TIME, ASSET_ID, *value_columns).sort([TIME, ASSET_ID])


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
