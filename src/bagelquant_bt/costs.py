"""Transaction cost and turnover calculations."""

from __future__ import annotations

import polars as pl

from .inputs import ASSET_ID, TIME


def turnover(weights: pl.DataFrame) -> pl.DataFrame:
    """Compute daily absolute weight turnover."""

    return (
        weights.sort([ASSET_ID, TIME])
        .with_columns(
            pl.col("weight")
            .fill_null(0.0)
            .shift(1)
            .over(ASSET_ID)
            .fill_null(0.0)
            .alias("previous_weight")
        )
        .with_columns(
            (pl.col("weight").fill_null(0.0) - pl.col("previous_weight"))
            .abs()
            .alias("weight_delta")
        )
        .group_by(TIME)
        .agg(pl.col("weight_delta").sum().alias("turnover"))
        .sort(TIME)
    )
