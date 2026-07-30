"""Return-series utilities for long-form Polars panels."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import polars as pl

from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME


@dataclass(frozen=True, slots=True)
class PreparedPriceData:
    """Observed prices, daily valuation prices, returns, and price-gap evidence."""

    observed_prices: pl.DataFrame
    valuation_prices: pl.DataFrame
    forward_returns: pl.DataFrame
    price_gaps: pl.DataFrame


def prepare_price_data(prices: pl.DataFrame) -> PreparedPriceData:
    """Build a shared daily valuation calendar without inventing tradable prices.

    The market calendar is the union of observed price dates.  Each asset is
    marked at its last observed price during a single-asset gap, so the gap has
    zero return and the cumulative move is recognized only on resumption.
    """

    observed_prices = prices.sort([TIME, ASSET_ID])
    sessions = observed_prices.select(TIME).unique().sort(TIME).with_row_index(
        "_session_index"
    )
    assets = observed_prices.select(ASSET_ID).unique()
    valuation_prices = (
        sessions.join(assets, how="cross")
        .join(observed_prices, on=[TIME, ASSET_ID], how="left")
        .with_columns(
            pl.col("price").alias("_observed_price"),
            pl.when(pl.col("price").is_not_null())
            .then(pl.col("_session_index"))
            .otherwise(None)
            .forward_fill()
            .over(ASSET_ID)
            .alias("_last_observed_session"),
            pl.when(pl.col("price").is_not_null())
            .then(pl.col(TIME))
            .otherwise(None)
            .forward_fill()
            .over(ASSET_ID)
            .alias("_last_observed_time"),
        )
        .with_columns(pl.col("price").forward_fill().over(ASSET_ID))
        .sort([ASSET_ID, TIME])
    )
    price_gaps = (
        valuation_prices.filter(
            pl.col("price").is_not_null() & pl.col("_observed_price").is_null()
        )
        .select(
            TIME,
            ASSET_ID,
            pl.col("_last_observed_time").alias("last_observed_time"),
            (pl.col("_session_index") - pl.col("_last_observed_session"))
            .cast(pl.Int64)
            .alias("missing_session_count"),
        )
        .sort([TIME, ASSET_ID])
    )
    forward_returns = (
        valuation_prices.with_columns(
            (pl.col("price").shift(-1).over(ASSET_ID) / pl.col("price") - 1.0).alias(
                "forward_return"
            )
        )
        .filter(pl.col("price").is_not_null() & pl.col("forward_return").is_not_null())
        .select(TIME, ASSET_ID, "forward_return")
        .sort([TIME, ASSET_ID])
    )
    return PreparedPriceData(
        observed_prices=observed_prices,
        valuation_prices=valuation_prices.select(TIME, ASSET_ID, "price"),
        forward_returns=forward_returns,
        price_gaps=price_gaps,
    )


def asset_close_to_close_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute daily close-to-close returns on the market session calendar."""

    return prepare_price_data(prices).forward_returns


def align_weights_to_forward_returns(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    price_data = prepare_price_data(prices)
    executable = _expand_portfolio_weights(
        weights,
        price_data.observed_prices,
        price_data.forward_returns,
    )
    return executable, price_data.forward_returns


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
        .sort([TIME, ASSET_ID])
        .group_by(TIME, maintain_order=True)
        .agg(pl.col("weighted_return").alias("_weighted_returns"))
        .with_columns(
            pl.col("_weighted_returns")
            .map_elements(math.fsum, return_dtype=pl.Float64)
            .alias("gross_return")
        )
        .drop("_weighted_returns")
        .sort(TIME)
    )


def _expand_portfolio_weights(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    initial_target_weights: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if weights.is_empty():
        return weights.select(TIME, ASSET_ID, "weight")
    if prices.is_empty():
        raise InputValidationError("weights has no covered price range")

    first_execution_time = weights.get_column(TIME).min()
    execution_keys = (
        forward_returns.select(TIME, ASSET_ID)
        .filter(pl.col(TIME) >= first_execution_time)
        .unique()
        .sort([ASSET_ID, TIME])
    )
    snapshot_times = (
        weights.select(TIME)
        .unique()
        .join(prices.select(TIME).unique(), on=TIME, how="inner")
    )
    snapshot_assets = prices.select(ASSET_ID).unique()
    target_snapshots = (
        snapshot_times.join(snapshot_assets, how="cross")
        .join(weights, on=[TIME, ASSET_ID], how="left")
        .with_columns(pl.col("weight").fill_null(0.0))
        .join(
            prices.select(TIME, ASSET_ID, pl.col("price").alias("_observed_price")),
            on=[TIME, ASSET_ID],
            how="left",
        )
        .sort([ASSET_ID, TIME])
    )
    tradable_targets = target_snapshots.filter(pl.col("_observed_price").is_not_null())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        expanded = execution_keys.join_asof(
            tradable_targets.select(TIME, ASSET_ID, "weight"),
            on=TIME,
            by=ASSET_ID,
            strategy="backward",
        )
    if initial_target_weights is not None:
        initial = initial_target_weights.select(
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("weight").cast(pl.Float64).alias("_initial_weight"),
        )
        expanded = (
            expanded.join(initial, on=ASSET_ID, how="left")
            .with_columns(
                pl.col("weight")
                .fill_null(pl.col("_initial_weight"))
                .alias("weight")
            )
            .drop("_initial_weight")
        )
    return (
        expanded
        .drop_nulls("weight")
        .select(TIME, ASSET_ID, "weight")
        .sort([TIME, ASSET_ID])
    )


def unexecuted_weight_keys(weights: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Describe rebalance targets blocked by an absent executable price."""

    if weights.is_empty() or prices.is_empty():
        return _empty_unexecuted_weight_keys()
    snapshot_times = (
        weights.select(TIME)
        .unique()
        .join(prices.select(TIME).unique(), on=TIME, how="inner")
    )
    snapshot_assets = prices.select(ASSET_ID).unique()
    target_snapshots = (
        snapshot_times.join(snapshot_assets, how="cross")
        .join(weights, on=[TIME, ASSET_ID], how="left")
        .with_columns(pl.col("weight").fill_null(0.0).alias("target_weight"))
        .drop("weight")
        .join(
            prices.select(TIME, ASSET_ID, pl.col("price").alias("_observed_price")),
            on=[TIME, ASSET_ID],
            how="left",
        )
        .sort([ASSET_ID, TIME])
    )
    tradable_targets = target_snapshots.filter(pl.col("_observed_price").is_not_null())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        prior_targets = target_snapshots.join_asof(
            tradable_targets.select(TIME, ASSET_ID, "target_weight"),
            on=TIME,
            by=ASSET_ID,
            strategy="backward",
            suffix="_executed",
        )
    return (
        prior_targets.filter(
            pl.col("_observed_price").is_null()
            & (
                pl.col("target_weight")
                != pl.col("target_weight_executed").fill_null(0.0)
            )
        )
        .select(
            TIME,
            ASSET_ID,
            "target_weight",
            pl.col("target_weight_executed").fill_null(0.0).alias("retained_weight"),
        )
        .sort([TIME, ASSET_ID])
    )


def _empty_unexecuted_weight_keys() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            TIME: pl.Date,
            ASSET_ID: pl.String,
            "target_weight": pl.Float64,
            "retained_weight": pl.Float64,
        }
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
