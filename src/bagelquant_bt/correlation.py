"""Bounded saved-value pair statistics with caller-proved daily continuation.

Callers own saved-input identity, date alignment, authorization and changed-date
proofs. State contains daily scalar summaries, never signal/holdings panels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from math import fsum, isfinite
from typing import Any, Literal

import polars as pl

from .factor_analysis import factor_return_correlation

PAIR_CORRELATION_POLICY = "bt.pair_correlation.v1"


def pair_correlation(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    kind: Literal["returns", "cross_section"] = "returns",
    method: Literal["pearson", "spearman"] = "pearson",
    changed_dates: Sequence[date],
    previous: Mapping[str, Any] | None = None,
    start: date,
    end: date,
    zero_missing: bool = False,
    minimum_samples: int = 12,
    minimum_assets: int = 3,
    check_canceled: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Update and summarize one bounded pair batch; see pair_correlation_batches."""
    return pair_correlation_batches(
        ((left, right, changed_dates),),
        kind=kind,
        method=method,
        previous=previous,
        start=start,
        end=end,
        zero_missing=zero_missing,
        minimum_samples=minimum_samples,
        minimum_assets=minimum_assets,
        check_canceled=check_canceled,
    )


def pair_correlation_batches(
    batches: Iterable[tuple[pl.DataFrame, pl.DataFrame, Sequence[date]]],
    *,
    kind: Literal["returns", "cross_section"] = "returns",
    method: Literal["pearson", "spearman"] = "pearson",
    previous: Mapping[str, Any] | None = None,
    start: date,
    end: date,
    zero_missing: bool = False,
    minimum_samples: int = 12,
    minimum_assets: int = 3,
    check_canceled: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Consume bounded pair batches and summarize the saved scalar history once.

    Each batch is ``(left, right, changed_dates)``. Changed dates must be unique
    across all batches; an empty batch iterable is a metadata-only continuation.
    Original daily state is copied once and never mutated, including on failure
    or cancellation. Frames are consumed one batch at a time without retention.
    Frames have Date ``time`` and numeric ``value``; cross-sections also require
    nonnull String ``asset_id``. Changed dates remove previous observations even
    when replacement inputs are empty. Spearman ranks average ties *after* the
    common finite-asset join. Daily coefficients receive equal weight.

    ``zero_missing`` is only for complete caller-proved holdings snapshots: an
    absent asset is unheld (zero), while an explicit null/nonfinite mark remains
    missing. The caller must exclude dates without complete account snapshots.
    Previous state must have the same mode/method/missingness policy; the caller
    must separately prove every retained date's original input identity.
    """
    cancel = check_canceled or (lambda: None)
    cancel()
    if kind not in {"returns", "cross_section"} or method not in {
        "pearson",
        "spearman",
    }:
        raise ValueError("Unsupported correlation kind or method")
    if kind == "returns" and (method != "pearson" or zero_missing):
        raise ValueError("Return correlation uses pairwise finite Pearson samples")
    if type(start) is not date or type(end) is not date or start > end:
        raise ValueError("Correlation requires ordered Date bounds")
    if any(
        type(value) is not int or value < 2
        for value in (minimum_samples, minimum_assets)
    ):
        raise ValueError("Correlation sample thresholds must be at least two")
    policy = {
        "policy": PAIR_CORRELATION_POLICY,
        "kind": kind,
        "method": method,
        "zero_missing": zero_missing,
        "minimum_samples": minimum_samples,
        "minimum_assets": minimum_assets,
    }
    if previous is not None and any(
        previous.get(key) != value for key, value in policy.items()
    ):
        raise ValueError("Previous correlation state has a different policy")
    # Retain scalar history once; replacement dates may represent deletions.
    days = {}
    for index, row in enumerate((previous or {}).get("days", ())):
        if index % 256 == 0:
            cancel()
        date.fromisoformat(row["time"])
        days[row["time"]] = dict(row)
    changed_all: set[date] = set()
    iterator = iter(batches)
    while True:
        cancel()
        try:
            left, right, changed_dates = next(iterator)
        except StopIteration:
            break
        cancel()
        changed = set(changed_dates)
        if (
            len(changed) != len(changed_dates)
            or any(type(day) is not date for day in changed)
            or not changed_all.isdisjoint(changed)
        ):
            raise ValueError("Changed dates must be unique Date values across batches")
        _validate_pair_frames(left, right, kind=kind, changed=changed)
        changed_all.update(changed)
        for day in changed:
            days.pop(day.isoformat(), None)
        _update_pair_days(
            days,
            left,
            right,
            kind=kind,
            method=method,
            zero_missing=zero_missing,
            minimum_assets=minimum_assets,
            check_canceled=cancel,
        )
        # Release consumed panels before asking a lazy producer for its next batch.
        del left, right, changed_dates
    cancel()
    return _summarize_pair_days(
        days,
        policy=policy,
        start=start,
        end=end,
        processed_dates=len(changed_all),
        check_canceled=cancel,
    )


def _validate_pair_frames(left, right, *, kind, changed):
    keys = ["time"] + (["asset_id"] if kind == "cross_section" else [])
    for frame in (left, right):
        if not {*keys, "value"} <= set(frame.columns):
            raise ValueError("Correlation inputs require typed coordinates and value")
        if frame.schema["time"] != pl.Date or frame["time"].null_count():
            raise ValueError("Correlation time must be nonnull Date")
        if not frame.schema["value"].is_numeric():
            raise ValueError("Correlation values must be numeric")
        if kind == "cross_section" and (
            frame.schema["asset_id"] != pl.String or frame["asset_id"].null_count()
        ):
            raise ValueError("Correlation assets must be nonnull String")
        if frame.select(keys).is_duplicated().any():
            raise ValueError("Correlation coordinates must be unique")
        if not set(frame["time"].to_list()) <= changed:
            raise ValueError("Correlation inputs exceed the admitted changed dates")


def _update_pair_days(
    days, left, right, *, kind, method, zero_missing, minimum_assets, check_canceled
):
    if kind == "returns":
        paired = (
            left.select("time", pl.col("value").alias("left"))
            .join(right.select("time", pl.col("value").alias("right")), on="time")
            .filter(pl.col("left").is_finite() & pl.col("right").is_finite())
            .sort("time")
        )
        for row in paired.iter_rows(named=True):
            check_canceled()
            days[str(row["time"])] = {**row, "time": str(row["time"])}
    else:
        keys = ["time", "asset_id"]
        left_frame = left.select(
            *keys, pl.col("value").alias("left"), pl.lit(True).alias("left_present")
        )
        right_frame = right.select(
            *keys, pl.col("value").alias("right"), pl.lit(True).alias("right_present")
        )
        paired = left_frame.join(
            right_frame, on=keys, how="full" if zero_missing else "inner", coalesce=True
        )
        if zero_missing:
            paired = paired.with_columns(
                pl.when(pl.col("left_present").is_null())
                .then(0.0)
                .otherwise(pl.col("left"))
                .alias("left"),
                pl.when(pl.col("right_present").is_null())
                .then(0.0)
                .otherwise(pl.col("right"))
                .alias("right"),
            )
            # Retained exit coordinates are unheld in both accounts and do
            # not belong to the held-asset union or its sample denominator.
            paired = paired.filter((pl.col("left") != 0) | (pl.col("right") != 0))
        paired = paired.filter(pl.col("left").is_finite() & pl.col("right").is_finite())
        x, y = pl.col("left"), pl.col("right")
        if method == "spearman":
            x, y = x.rank(method="average"), y.rank(method="average")
        grouped = (
            paired.group_by("time")
            .agg(
                pl.len().alias("sample_count"),
                pl.col("left").n_unique().alias("left_unique"),
                pl.col("right").n_unique().alias("right_unique"),
                pl.corr(x, y).alias("correlation"),
            )
            .sort("time")
        )
        for row in grouped.iter_rows(named=True):
            check_canceled()
            valid = (
                row["sample_count"] >= minimum_assets
                and row["left_unique"] >= 2
                and row["right_unique"] >= 2
                and row["correlation"] is not None
                and isfinite(row["correlation"])
            )
            days[str(row["time"])] = {
                "time": str(row["time"]),
                "sample_count": row["sample_count"],
                "correlation": float(max(-1.0, min(1.0, row["correlation"])))
                if valid
                else None,
            }


def _summarize_pair_days(days, *, policy, start, end, processed_dates, check_canceled):
    ordered = sorted(days.items())
    start_key, end_key = start.isoformat(), end.isoformat()
    selected = [row for day, row in ordered if start_key <= day <= end_key]
    if policy["kind"] == "returns":
        left_frame = pl.DataFrame(
            {
                "time": [date.fromisoformat(row["time"]) for row in selected],
                "value": [row["left"] for row in selected],
            },
            schema={"time": pl.Date, "value": pl.Float64},
        )
        right_frame = pl.DataFrame(
            {"time": left_frame["time"], "value": [row["right"] for row in selected]},
            schema={"time": pl.Date, "value": pl.Float64},
        )
        result = factor_return_correlation(
            left_frame, right_frame, minimum_samples=policy["minimum_samples"]
        )
        if result["correlation"] is not None and not isfinite(result["correlation"]):
            result.update(correlation=None, reason="undefined_correlation")
    else:
        valid_days = [row for row in selected if row["correlation"] is not None]
        result = {
            "correlation": fsum(row["correlation"] for row in valid_days)
            / len(valid_days)
            if valid_days
            else None,
            "sample_count": len(valid_days),
            "start_date": valid_days[0]["time"] if valid_days else None,
            "end_date": valid_days[-1]["time"] if valid_days else None,
            "asset_sample_count": sum(row["sample_count"] for row in valid_days),
            "reason": None if valid_days else "insufficient_cross_sections",
        }
    check_canceled()
    return {
        **result,
        "state": {**policy, "days": [row for _, row in ordered]},
        "processed_dates": processed_dates,
    }
