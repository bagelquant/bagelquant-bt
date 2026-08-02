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


def validate_universe(universe: pl.DataFrame) -> pl.DataFrame:
    """Validate a membership panel keyed by time and asset identifier."""

    if not isinstance(universe, pl.DataFrame):
        raise InputValidationError("coverage_universe must be a polars DataFrame")
    missing = sorted({TIME, ASSET_ID} - set(universe.columns))
    if missing:
        raise InputValidationError(
            f"coverage_universe is missing required columns: {missing}"
        )
    normalized = universe.select(TIME, ASSET_ID).with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
    ).drop_nulls([TIME, ASSET_ID])
    if normalized.select(pl.struct(TIME, ASSET_ID).is_duplicated().any()).item():
        raise InputValidationError(
            "coverage_universe must be unique by (time, asset_id)"
        )
    return normalized.sort([TIME, ASSET_ID])


def validate_execution_availability(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate sparse, caller-authored market execution constraints."""

    if not isinstance(frame, pl.DataFrame):
        raise InputValidationError(
            "execution_availability must be a polars DataFrame"
        )
    required = {TIME, ASSET_ID, "can_buy", "can_sell", "reason"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError(
            f"execution_availability is missing required columns: {missing}"
        )
    normalized = frame.select(
        TIME, ASSET_ID, "can_buy", "can_sell", "reason"
    ).with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("can_buy").cast(pl.Boolean, strict=False),
        pl.col("can_sell").cast(pl.Boolean, strict=False),
        pl.col("reason").cast(pl.String),
    )
    normalized = normalized.drop_nulls(
        [TIME, ASSET_ID, "can_buy", "can_sell", "reason"]
    )
    if normalized.select(pl.struct(TIME, ASSET_ID).is_duplicated().any()).item():
        raise InputValidationError(
            "execution_availability must be unique by (time, asset_id)"
        )
    return normalized.sort([TIME, ASSET_ID])


def validate_slippage_rates(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate sparse effective-dated per-asset slippage rates."""

    if not isinstance(frame, pl.DataFrame):
        raise InputValidationError("slippage_rates must be a polars DataFrame")
    required = {TIME, ASSET_ID, "slippage_rate"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError(
            f"slippage_rates is missing required columns: {missing}"
        )
    expressions = [
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("slippage_rate").cast(pl.Float64, strict=False),
    ]
    columns = [TIME, ASSET_ID, "slippage_rate"]
    if "is_fallback" in frame.columns:
        columns.append("is_fallback")
        expressions.append(pl.col("is_fallback").cast(pl.Boolean, strict=False))
    normalized = frame.select(columns).with_columns(*expressions)
    required_values = [TIME, ASSET_ID, "slippage_rate"]
    if "is_fallback" in columns:
        required_values.append("is_fallback")
    if normalized.select(
        pl.any_horizontal(pl.col(column).is_null() for column in required_values).any()
    ).item():
        raise InputValidationError("slippage_rates must not contain null values")
    if normalized.select(
        ((~pl.col("slippage_rate").is_finite()) | (pl.col("slippage_rate") < 0)).any()
    ).item():
        raise InputValidationError(
            "slippage_rates.slippage_rate must be finite and nonnegative"
        )
    if normalized.select(pl.struct(TIME, ASSET_ID).is_duplicated().any()).item():
        raise InputValidationError(
            "slippage_rates must be unique by (time, asset_id)"
        )
    return normalized.sort([ASSET_ID, TIME])


def missing_price_keys(frame: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Return frame keys without an exact matching price key."""

    return (
        frame.select(TIME, ASSET_ID)
        .join(prices.select(TIME, ASSET_ID), on=[TIME, ASSET_ID], how="anti")
        .sort([TIME, ASSET_ID])
    )


def asset_coverage(
    frame: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    asset_count_column: str,
    coverage_universe: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Count raw input assets against a supplied membership or price universe."""

    universe_keys = (
        validate_universe(coverage_universe)
        if coverage_universe is not None
        else prices.select(TIME, ASSET_ID)
    )
    universe = universe_keys.group_by(TIME).agg(
        pl.len().alias("universe_asset_count")
    ).sort(TIME)
    input_counts = frame.group_by(TIME).agg(pl.len().alias(asset_count_column))
    return (
        universe.join(input_counts, on=TIME, how="left")
        .with_columns(
            pl.col(asset_count_column).fill_null(0).cast(pl.Int64),
            pl.col("universe_asset_count").cast(pl.Int64),
        )
        .with_columns(
            (pl.col(asset_count_column) / pl.col("universe_asset_count")).alias(
                "coverage_ratio"
            )
        )
        .select(TIME, asset_count_column, "universe_asset_count", "coverage_ratio")
        .sort(TIME)
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
