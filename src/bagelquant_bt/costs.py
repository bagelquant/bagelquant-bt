"""Transaction cost and turnover calculations."""

from __future__ import annotations

import polars as pl

from .inputs import ASSET_ID, TIME


def turnover(
    weights: pl.DataFrame,
    *,
    initial_weights: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute daily absolute weight turnover."""

    return _turnover_from_weight_deltas(
        _weight_deltas(weights, initial_weights=initial_weights)
    )


def _turnover_from_weight_deltas(deltas: pl.DataFrame) -> pl.DataFrame:
    """Aggregate a previously computed per-asset weight-delta frame."""

    return (
        deltas.group_by(TIME)
        .agg(pl.col("weight_delta").sum().alias("turnover"))
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
    return _weight_delta_plan(weights, initial_weights=initial_weights).collect(
        engine="auto"
    )


def _weight_delta_plan(
    weights: pl.DataFrame,
    *,
    initial_weights: pl.DataFrame | None,
) -> pl.LazyFrame:
    initial = (
        pl.DataFrame(schema={ASSET_ID: pl.String, "_initial_weight": pl.Float64})
        if initial_weights is None
        else initial_weights.select(
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("weight").cast(pl.Float64).alias("_initial_weight"),
        )
    )
    return (
        weights.lazy()
        .sort([ASSET_ID, TIME])
        .join(initial.lazy(), on=ASSET_ID, how="left")
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
            (
                pl.col("weight").fill_null(0.0)
                - pl.col("previous_weight")
            ).alias("signed_weight_delta")
        )
        .with_columns(
            pl.col("signed_weight_delta").abs().alias("weight_delta")
        )
        .drop("_initial_weight")
    )


def _sparse_weight_deltas(
    weights: pl.DataFrame,
    *,
    initial_weights: pl.DataFrame | None,
) -> pl.DataFrame:
    """Return only non-zero weight changes for internal economic calculations."""

    return (
        _weight_delta_plan(weights, initial_weights=initial_weights)
        .filter(pl.col("weight_delta") != 0.0)
        .collect(engine="auto")
    )
