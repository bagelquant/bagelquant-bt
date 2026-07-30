"""Transaction cost and turnover calculations."""

from __future__ import annotations

import math

import polars as pl

from .inputs import ASSET_ID, TIME


def turnover(
    weights: pl.DataFrame,
    *,
    initial_weights: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute daily absolute weight turnover."""

    return (
        _weight_deltas(weights, initial_weights=initial_weights)
        .sort([TIME, ASSET_ID])
        .group_by(TIME, maintain_order=True)
        .agg(pl.col("weight_delta").alias("_weight_deltas"))
        .with_columns(
            pl.col("_weight_deltas")
            .map_elements(math.fsum, return_dtype=pl.Float64)
            .alias("turnover")
        )
        .drop("_weight_deltas")
        .sort(TIME)
    )


def weight_deltas(
    weights: pl.DataFrame,
    *,
    initial_weights: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return per-asset deltas, optionally continuing from a checkpoint."""

    return _weight_deltas(weights, initial_weights=initial_weights)


def _weight_deltas(
    weights: pl.DataFrame,
    *,
    initial_weights: pl.DataFrame | None,
) -> pl.DataFrame:
    initial = (
        pl.DataFrame(schema={ASSET_ID: pl.String, "_initial_weight": pl.Float64})
        if initial_weights is None
        else initial_weights.select(
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("weight").cast(pl.Float64).alias("_initial_weight"),
        )
    )
    return (
        weights.sort([ASSET_ID, TIME])
        .join(initial, on=ASSET_ID, how="left")
        .with_columns(
            pl.col("weight")
            .fill_null(0.0)
            .shift(1)
            .over(ASSET_ID)
            .fill_null(pl.col("_initial_weight"))
            .fill_null(0.0)
            .alias("previous_weight")
        )
        .with_columns(
            (pl.col("weight").fill_null(0.0) - pl.col("previous_weight"))
            .abs()
            .alias("weight_delta")
        )
        .drop("_initial_weight")
    )
