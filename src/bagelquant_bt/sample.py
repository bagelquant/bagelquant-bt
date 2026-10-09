"""Effective saved-evaluation samples, independent of returns' direction."""

from collections.abc import Mapping
from datetime import date

import polars as pl

SAMPLE_POLICY = "evaluation.effective_sample.v1"


def evaluation_sample(
    frames: Mapping[str, pl.DataFrame],
    *,
    start: date,
    end: date,
    minimum_history_years: int | None = None,
) -> dict:
    """Describe the requested interval after excluding leading unavailable history.

    Signal coverage supplies the observation calendar and the first finite signal.
    Without it, finite saved economic observations supply the start. Zero returns
    are valid. Later missing observations never move the original sample start.
    """
    if start > end:
        raise ValueError("sample start must not follow end")
    if minimum_history_years is not None and (
        type(minimum_history_years) is not int or minimum_history_years < 0
    ):
        raise ValueError(
            "minimum history must be a nonnegative number of calendar years"
        )
    coverage = frames.get("daily_signal_coverage", pl.DataFrame())
    signal_start = None
    if {"time", "signal_count"} <= set(coverage.columns):
        available = coverage.filter(pl.col("signal_count") > 0)
        signal_start = available["time"].min() if not available.is_empty() else None
    candidates = []
    return_starts = []
    fallback_calendar = pl.DataFrame()
    for name, column in (
        ("daily_book_returns", "net_return"),
        ("daily_tail_returns", "net_return"),
        ("horizon_ic", "spearman_ic"),
        ("horizon_book_returns", "book_return"),
        ("daily_signal_autocorrelation", "rank_autocorrelation"),
    ):
        frame = frames.get(name, pl.DataFrame())
        coordinate = "evaluation_date" if "evaluation_date" in frame.columns else "time"
        if {coordinate, column} <= set(frame.columns):
            for maturity in (
                "target_end_date",
                "interval_end",
                "available_date",
                "target_date",
            ):
                if maturity in frame.columns:
                    frame = frame.filter(pl.col(maturity) <= end)
            finite = frame.filter(pl.col(column).is_finite())
            if not finite.is_empty():
                candidates.append(finite[coordinate].min())
                if name in {"daily_book_returns", "daily_tail_returns"}:
                    return_starts.append(finite[coordinate].min())
                if fallback_calendar.is_empty():
                    fallback_calendar = frame.select(
                        pl.col(coordinate).alias("time"),
                        pl.col(column)
                        .is_finite()
                        .fill_null(False)
                        .cast(pl.Int64)
                        .alias("signal_count"),
                    )
                    fallback_calendar = (
                        fallback_calendar.group_by("time")
                        .agg(pl.col("signal_count").max())
                        .sort("time")
                    )
    observation_start = (
        signal_start
        if "signal_count" in coverage.columns
        else min(candidates, default=None)
    )
    effective_start = max(start, observation_start) if observation_start else None
    effective_start = (
        effective_start if effective_start and effective_start <= end else None
    )
    calendar = (
        coverage.filter(pl.col("time").is_between(start, end))
        if "time" in coverage.columns
        else fallback_calendar.filter(pl.col("time").is_between(start, end))
        if "time" in fallback_calendar.columns
        else pl.DataFrame()
    )
    selected = (
        calendar.filter(pl.col("time") >= effective_start)
        if effective_start and not calendar.is_empty()
        else pl.DataFrame()
    )
    years = (end - effective_start).days / 365.2425 if effective_start else 0.0
    valid = (
        int((selected["signal_count"] > 0).sum())
        if "signal_count" in selected.columns
        else None
    )
    warnings = []
    if effective_start is None:
        warnings.append("no_available_observations")
    elif minimum_history_years is not None:
        try:
            anniversary = effective_start.replace(
                year=effective_start.year + minimum_history_years
            )
        except ValueError:
            anniversary = effective_start.replace(
                year=effective_start.year + minimum_history_years, day=28
            )
        if end < anniversary:
            warnings.append("effective_history_below_minimum")
    return {
        "policy": SAMPLE_POLICY,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "effective_start": effective_start.isoformat() if effective_start else None,
        "effective_end": end.isoformat() if effective_start else None,
        "first_signal_date": signal_start.isoformat() if signal_start else None,
        "first_return_date": min(return_starts).isoformat() if return_starts else None,
        "calendar_basis": "selected_universe"
        if "time" in coverage.columns
        else "saved_observations",
        "effective_years": years,
        "expected_observations": selected.height if not calendar.is_empty() else None,
        "valid_observations": valid,
        "excluded_leading_observations": calendar.height - selected.height
        if not calendar.is_empty()
        else None,
        "missing_ratio": 1 - valid / selected.height
        if valid is not None and selected.height
        else None,
        "warnings": warnings,
    }
