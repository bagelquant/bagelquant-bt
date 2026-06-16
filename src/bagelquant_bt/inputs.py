"""Input validation for long-form Polars backtest data."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from .exceptions import InputValidationError

TIME = "time"
ASSET_ID = "asset_id"


def validate_panel_frame(
    frame: pl.DataFrame,
    *,
    label: str,
    value_columns: Iterable[str],
) -> pl.DataFrame:
    """Validate a long-form panel and return a defensive sorted clone."""

    if not isinstance(frame, pl.DataFrame):
        raise InputValidationError(f"{label} must be a polars DataFrame")
    columns = set(frame.columns)
    required = {TIME, ASSET_ID, *value_columns}
    missing = sorted(required - columns)
    if missing:
        raise InputValidationError(f"{label} is missing required columns: {missing}")
    normalized = frame.clone().with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
    )
    for column in value_columns:
        if not normalized.schema[column].is_numeric():
            raise InputValidationError(f"{label}.{column} must be numeric")
    normalized = normalized.drop_nulls([TIME, ASSET_ID, *value_columns])
    for column in value_columns:
        normalized = normalized.filter(~pl.col(column).is_nan())
    if normalized.select(pl.struct(TIME, ASSET_ID).is_duplicated().any()).item():
        raise InputValidationError(f"{label} must be unique by (time, asset_id)")
    return normalized.sort([TIME, ASSET_ID])


def validate_prices(prices: pl.DataFrame) -> pl.DataFrame:
    return validate_panel_frame(prices, label="prices", value_columns=("price",))


def validate_weights(weights: pl.DataFrame) -> pl.DataFrame:
    return validate_panel_frame(weights, label="weights", value_columns=("weight",))


def validate_factor(factor: pl.DataFrame) -> pl.DataFrame:
    return validate_panel_frame(factor, label="factor", value_columns=("factor",))


def missing_price_keys(frame: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Return frame keys without an exact matching price key."""

    return (
        frame.select(TIME, ASSET_ID)
        .join(prices.select(TIME, ASSET_ID), on=[TIME, ASSET_ID], how="anti")
        .sort([TIME, ASSET_ID])
    )


def align_signal_and_prices(
    signal: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    signal_column: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep only overlapping (time, asset_id) rows for a signal and prices."""

    signal_frame = validate_panel_frame(
        signal,
        label="signal",
        value_columns=(signal_column,),
    )
    price_frame = validate_prices(prices)
    keys = price_frame.select(TIME, ASSET_ID).join(
        signal_frame.select(TIME, ASSET_ID),
        on=[TIME, ASSET_ID],
        how="inner",
    )
    return (
        signal_frame.join(keys, on=[TIME, ASSET_ID], how="inner").sort(
            [TIME, ASSET_ID]
        ),
        price_frame.join(keys, on=[TIME, ASSET_ID], how="inner").sort([TIME, ASSET_ID]),
    )
