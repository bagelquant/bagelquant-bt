"""Internal helpers for deterministic numeric Quantile ordering."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def quantile_number(label: object) -> int:
    """Return the positive numeric suffix from one canonical ``qN`` label."""

    text = str(label)
    if not text.startswith("q") or not text[1:].isdigit() or int(text[1:]) < 1:
        raise ValueError(f"invalid quantile label: {text}")
    return int(text[1:])


def ordered_quantile_labels(labels: Iterable[object]) -> list[str]:
    """Return unique canonical labels in numeric q1-to-qN order."""

    return sorted({str(label) for label in labels}, key=quantile_number)


def sort_quantile_frame(
    frame: pl.DataFrame,
    *,
    before: tuple[str, ...] = (),
    after: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Sort a Quantile frame without lexicographically placing q10 after q1."""

    if frame.is_empty() or "quantile" not in frame.columns:
        columns = [*before, *after]
        return frame.sort(columns) if columns else frame
    ordered_quantile_labels(frame.get_column("quantile").drop_nulls().to_list())
    order_column = "__quantile_number"
    return (
        frame.with_columns(
            pl.col("quantile")
            .str.slice(1)
            .cast(pl.Int64)
            .alias(order_column)
        )
        .sort([*before, order_column, *after])
        .drop(order_column)
    )


__all__ = ["ordered_quantile_labels", "quantile_number", "sort_quantile_frame"]
