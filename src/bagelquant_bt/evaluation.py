"""Standalone evaluation of saved Alpha values and Portfolio weights."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any

import polars as pl
from bagelquant_core import Node
from bagelquant_core.hashing import hash_mapping

from .exceptions import InputValidationError
from .horizon import (
    DAILY_ALPHA_RETURN_LAGS,
    DAILY_BOOK_LEAD_LAGS,
    DAILY_SESSION_WINDOWS,
    DAILY_SUMMARY_AUTOCORRELATION_LAGS,
    SessionWindow,
    _centered_rank_book_weights_prepared,
    _gross_one_tail_weights_with_membership,
    _prepare_factor_returns,
    _quantile_membership,
    _validate_scheduled_factor_frame,
    _window_factor_returns_prepared,
    _window_quantile_forward_returns_with_membership,
    build_statistical_inference,
    quantile_curve_structure,
    rolling_window_information_coefficients,
    signal_rank_persistence,
    summarize_window_ic,
    window_book_returns,
    window_information_coefficients,
    window_tail_returns,
)
from .portfolio_mechanics import (
    WeightStateCheckpoint,
    node_values,
    normalize_forward_returns,
    select_rebalance_snapshots,
    session_calendar,
    simulate_weight_returns,
)
from .runtime import BTExecutionOptions, run_evaluation_batch
from .statistics import return_statistics

_ALPHA_COMPONENTS = frozenset((
    "horizons", "rolling_ic", "book_tail", "turnover", "lead_lag",
    "persistence", "alpha_return", "quantiles",
))


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Tables/metrics, optional weight state and identity-neutral execution evidence."""

    tables: dict[str, pl.DataFrame]
    metrics: dict[str, Any]
    checkpoint: WeightStateCheckpoint | None = None
    execution: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)


def _lags(values: Sequence[int], *, negative: bool = False) -> tuple[int, ...]:
    if (
        not values
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or (value < 0 and not negative)
            for value in values
        )
        or len(set(values)) != len(values)
    ):
        raise InputValidationError("lags must be unique integers in the allowed range")
    return tuple(sorted(values))


def _shift_weights(
    weights: pl.DataFrame, calendar: pl.DataFrame, lag: int
) -> pl.DataFrame:
    dates = calendar.with_row_index("_ordinal")
    mapped = dates.with_columns(
        (pl.col("_ordinal").cast(pl.Int64) - lag).alias("_source")
    )
    source = dates.select(
        pl.col("_ordinal").cast(pl.Int64).alias("_source"),
        pl.col("time").alias("source_time"),
    )
    return (
        mapped.join(source, on="_source", how="inner")
        .select("time", "source_time")
        .join(weights.rename({"time": "source_time"}), on="source_time", how="inner")
        .select("time", "asset_id", "value")
        .sort("time", "asset_id")
    )


def evaluate_weights(
    weights: Node,
    forward_returns: pl.DataFrame,
    *,
    calendar: pl.DataFrame | None = None,
    anchor: date | None = None,
    every: int = 1,
    decisions: pl.DataFrame | None = None,
    cost_rate: float = 0.0005,
    annualization: int = 252,
    available_date: date | None = None,
    lags: Sequence[int] = DAILY_ALPHA_RETURN_LAGS,
    components: Sequence[str] | None = None,
    checkpoint: WeightStateCheckpoint | None = None,
    check_canceled: Callable[[], None] | None = None,
) -> EvaluationResult:
    """Evaluate explicit weights using caller-aligned forward returns.

    Each provided snapshot is a complete target. Holdings drift between
    rebalances. Net turnover is the full absolute change from its own drifted
    pretrade weights; Net return subtracts ``cost_rate * turnover`` directly.
    Gross and Net holdings subsequently drift independently. An anchored
    cadence selects the latest whole snapshot, never a future snapshot.
    """

    selected = set(("returns", "lags") if components is None else components)
    if not selected or selected - {"returns", "lags"}:
        raise InputValidationError(f"unknown weight components: {sorted(selected)}")
    values = node_values(weights, kind="weights")
    if values.is_empty() and checkpoint is None:
        raise InputValidationError(
            "weights requires at least one finite target snapshot"
        )
    sessions = session_calendar(calendar, default=weights.domain.times)
    labels = normalize_forward_returns(
        forward_returns, calendar=sessions, available_date=available_date
    )
    cutoff = (
        available_date if available_date is not None else labels["available_date"].max()
    )
    if (
        checkpoint is not None
        and cutoff is not None
        and checkpoint.available_date > cutoff
    ):
        raise InputValidationError(
            "checkpoint includes information beyond available_date"
        )
    if every != 1:
        if calendar is None or anchor is None or decisions is not None:
            raise InputValidationError(
                "cadence requires calendar and anchor without explicit decisions"
            )
        values = select_rebalance_snapshots(
            values, calendar=sessions, every=every, anchor=anchor
        )
    elif isinstance(every, bool) or not isinstance(every, int):
        raise InputValidationError("every must be a positive integer")
    if check_canceled:
        check_canceled()
    tables, final_checkpoint = simulate_weight_returns(
        values,
        labels,
        decisions=decisions,
        cost_rate=cost_rate,
        checkpoint=checkpoint,
        check_canceled=check_canceled,
    )
    metrics: dict[str, Any] = return_statistics(
        tables["returns"], annualization=annualization
    )
    metrics["average_turnover"] = tables["turnover"]["turnover"].mean()
    metrics["total_cost_return"] = float(tables["returns"]["cost_return"].sum())
    if "lags" in selected:
        if checkpoint is not None:
            raise InputValidationError(
                "lag diagnostics require complete weights without a checkpoint suffix"
            )
        lag_frames = []
        for lag in _lags(lags):
            if check_canceled:
                check_canceled()
            shifted = _shift_weights(values, sessions, lag)
            frame, _ = simulate_weight_returns(
                shifted,
                labels,
                decisions=None,
                cost_rate=cost_rate,
                include_holdings=False,
                check_canceled=check_canceled,
            )
            lag_frames.append(
                frame["returns"].with_columns(pl.lit(lag, dtype=pl.Int64).alias("lag"))
            )
        common = _common_lag_dates(lag_frames)
        tables["lag_returns"] = (
            pl.concat(lag_frames)
            .join(common, on="time", how="inner")
            .sort("lag", "time")
        )
        metrics["lags"] = [
            {
                "lag": lag,
                **return_statistics(
                    tables["lag_returns"].filter(pl.col("lag") == lag),
                    annualization=annualization,
                ),
            }
            for lag in _lags(lags)
        ]
    return EvaluationResult(tables, metrics, final_checkpoint)


def _common_lag_dates(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    common = frames[0].select("time").unique()
    for frame in frames[1:]:
        common = common.join(frame.select("time").unique(), on="time", how="inner")
    return common.sort("time")


_WINDOW_COLUMNS = ("window_kind", "window_id", "start_session", "end_session")
_GROUP_COLUMNS = (
    "evaluation_date",
    "execution_date",
    "target_end_date",
    *_WINDOW_COLUMNS,
)


def _window_labels(
    factor: pl.DataFrame,
    labels: pl.DataFrame,
    calendar: pl.DataFrame,
    window: SessionWindow,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compound one bounded label window with explicit maturity and coverage.

    Rolling reductions retain one panel at a time, rather than expanding each
    row by all future offsets. Missing observations invalidate that member's
    entire label; rank portfolios preserve its weight and contribute zero.
    """

    dates = calendar.with_row_index("_ordinal").with_columns(
        pl.col("_ordinal").cast(pl.Int64)
    )
    width = window.width
    source = labels.join(dates, on="time", how="inner").sort("asset_id", "_ordinal")
    source = (
        source.with_columns(
            pl.when(pl.col("forward_return") > -1)
            .then(pl.col("forward_return").log1p())
            .otherwise(0.0)
            .alias("_log_return"),
            (pl.col("forward_return") == -1)
            .fill_null(False)
            .cast(pl.Int64)
            .alias("_zero_wealth"),
            pl.col("forward_return").is_not_null().cast(pl.Int64).alias("_observed"),
        )
        .with_columns(
            pl.col("_log_return")
            .rolling_sum(window_size=width)
            .over("asset_id")
            .alias("_log_total"),
            pl.col("_zero_wealth")
            .rolling_sum(window_size=width)
            .over("asset_id")
            .alias("_zero_count"),
            pl.col("_observed")
            .rolling_sum(window_size=width)
            .over("asset_id")
            .alias("_observation_count"),
            pl.col("_ordinal")
            .shift(width - 1)
            .over("asset_id")
            .alias("_first_ordinal"),
        )
        .with_columns(
            pl.when(
                (pl.col("_observation_count") == width)
                & (pl.col("_ordinal") - pl.col("_first_ordinal") == width - 1)
            )
            .then(
                pl.when(pl.col("_zero_count") > 0)
                .then(-1.0)
                .otherwise(pl.col("_log_total").exp() - 1.0)
            )
            .otherwise(None)
            .alias("_window_return"),
            (pl.col("_ordinal") - (window.end_session - 1)).alias(
                "_evaluation_ordinal"
            ),
        )
    )
    timing = (
        labels.group_by("time")
        .agg(
            pl.first("interval_start").alias("_interval_start"),
            pl.first("interval_end").alias("target_end_date"),
            pl.col("available_date").max().alias("_availability"),
            pl.col("is_mature").all().cast(pl.Int64).alias("_mature"),
        )
        .join(dates, on="time", how="inner")
        .sort("_ordinal")
        .with_columns(
            (
                pl.col("target_end_date").shift(1).is_not_null()
                & (pl.col("_interval_start") != pl.col("target_end_date").shift(1))
            )
            .cast(pl.Int64)
            .alias("_gap")
        )
        .with_columns(
            pl.col("_mature").rolling_sum(window_size=width).alias("_mature_count"),
            pl.col("_ordinal").shift(width - 1).alias("_first_ordinal"),
            pl.col("_interval_start").shift(width - 1).alias("target_start_date"),
            pl.col("_availability")
            .cast(pl.Int64)
            .rolling_max(window_size=width)
            .cast(pl.Date)
            .alias("available_date"),
            (
                pl.col("_gap").rolling_sum(window_size=width - 1)
                if width > 1
                else pl.lit(0)
            ).alias("_gap_count"),
        )
        .with_columns(
            (
                (pl.col("_mature_count") == width)
                & (pl.col("_ordinal") - pl.col("_first_ordinal") == width - 1)
            )
            .fill_null(False)
            .alias("_window_mature"),
            (pl.col("_gap_count") == 0).fill_null(False).alias("_window_contiguous"),
            (pl.col("_ordinal") - (window.end_session - 1)).alias(
                "_evaluation_ordinal"
            ),
        )
    )
    starts = (
        factor.select("evaluation_date", "execution_date", "asset_id")
        .join(
            dates.rename(
                {"time": "evaluation_date", "_ordinal": "_evaluation_ordinal"}
            ),
            on="evaluation_date",
            how="inner",
        )
        .join(
            timing.select(
                "_evaluation_ordinal",
                "target_start_date",
                "target_end_date",
                "available_date",
                "_window_mature",
                "_window_contiguous",
            ),
            on="_evaluation_ordinal",
            how="left",
        )
        .join(
            source.select("_evaluation_ordinal", "asset_id", "_window_return"),
            on=["_evaluation_ordinal", "asset_id"],
            how="left",
        )
        .with_columns(
            pl.lit(window.window_kind).alias("window_kind"),
            pl.lit(window.window_id).alias("window_id"),
            pl.lit(window.start_session, dtype=pl.Int64).alias("start_session"),
            pl.lit(window.end_session, dtype=pl.Int64).alias("end_session"),
            pl.col("_window_mature").fill_null(False),
            pl.col("_window_contiguous").fill_null(False),
            pl.col("target_start_date").alias("execution_date"),
        )
        .with_columns(
            (pl.col("_window_mature") & pl.col("_window_contiguous")).alias(
                "_window_usable"
            )
        )
    )
    coverage = (
        starts.group_by(*_GROUP_COLUMNS)
        .agg(
            pl.len().cast(pl.Int64).alias("expected_count"),
            (pl.col("_window_return").is_not_null() & pl.col("_window_usable"))
            .sum()
            .cast(pl.Int64)
            .alias("observed_count"),
            pl.col("_window_mature").all().alias("is_mature"),
            pl.col("_window_contiguous").all().alias("is_complete"),
            pl.col("available_date").max(),
        )
        .with_columns(
            (pl.col("observed_count") / pl.col("expected_count")).alias(
                "coverage_ratio"
            ),
            pl.when(~pl.col("is_mature"))
            .then(pl.lit("return window is immature"))
            .when(~pl.col("is_complete"))
            .then(pl.lit("economic intervals are not contiguous"))
            .when(pl.col("observed_count") != pl.col("expected_count"))
            .then(pl.lit("forward-return coverage is incomplete"))
            .otherwise(None)
            .alias("unavailable_reason"),
        )
        .sort("evaluation_date")
        .with_columns(pl.col("execution_date").alias("target_start_date"))
    )
    mature = (
        starts.filter("_window_usable")
        .select(
            *_GROUP_COLUMNS,
            "asset_id",
            "target_start_date",
            "available_date",
            pl.col("_window_return").alias("forward_return"),
        )
        .sort("evaluation_date", "asset_id")
    )
    return mature, coverage


def _path_frames(
    weights: pl.DataFrame,
    labels: pl.DataFrame,
    calendar: pl.DataFrame,
    lags: Sequence[int],
    cost_rate: float,
    check_canceled: Callable[[], None] | None,
    *,
    path_runner=None,
    path_id="lag",
    previous=None,
) -> pl.DataFrame:
    frames = []
    for lag in lags:
        if check_canceled:
            check_canceled()
        shifted = _shift_weights(weights, calendar, lag)
        if path_runner is None:
            tables, _ = simulate_weight_returns(
                shifted,
                labels,
                decisions=None,
                cost_rate=cost_rate,
                include_holdings=False,
                check_canceled=check_canceled,
            )
        else:
            tables = path_runner(
                f"{path_id}_{lag}", shifted, cost_rate, end=shifted["time"].max()
            )
        frame = (
            tables["returns"]
            .join(shifted.select("time").unique(), on="time", how="inner")
            .with_columns(pl.lit(lag, dtype=pl.Int64).alias("lag"))
        )
        if previous is not None:
            prior = previous.filter(pl.col("lag") == lag).join(
                frame.select("time").unique(), on="time", how="anti"
            )
            frame = pl.concat([prior, frame], how="diagonal_relaxed").sort("time")
        frames.append(frame)
    return (
        pl.concat(frames)
        .join(_common_lag_dates(frames), on="time", how="inner")
        .sort("lag", "time")
    )


def _alpha_window_frames(
    window: SessionWindow,
    *,
    factor: pl.DataFrame,
    labels: pl.DataFrame,
    sessions: pl.DataFrame,
    book: pl.DataFrame,
    tail: pl.DataFrame,
    quantiles: int,
    factor_standardization: str,
    quantile_membership: pl.DataFrame,
    prepared_factor: pl.DataFrame,
    check_canceled: Callable[[], None],
    previous: dict[str, pl.DataFrame] | None = None,
) -> dict[str, pl.DataFrame]:
    """Compute one window with private frames and shared read-only inputs."""
    check_canceled()
    previous = previous or {}
    old_coverage = previous.get("coverage")
    if old_coverage is not None:
        completed = old_coverage.filter("is_mature").select("evaluation_date").unique()
        factor = factor.join(completed, on="evaluation_date", how="anti")
        keys = factor.select("evaluation_date", "asset_id").unique()
        book = book.join(keys, on=["evaluation_date", "asset_id"], how="semi")
        tail = tail.join(keys, on=["evaluation_date", "asset_id"], how="semi")
        quantile_membership = quantile_membership.join(
            keys, on=["evaluation_date", "asset_id"], how="semi"
        )
        prepared_factor = prepared_factor.join(
            keys, on=["evaluation_date", "asset_id"], how="semi"
        )
        if not factor.is_empty():
            lower = factor["evaluation_date"].min()
            # Keep only label observations consumed by this window. Session
            # ordinals still come from the complete admitted calendar.
            labels = labels.filter(pl.col("time") >= lower)
    window_labels, coverage = _window_labels(factor, labels, sessions, window)
    check_canceled()
    ic = window_information_coefficients(factor, window_labels)
    check_canceled()
    quantile_returns = _window_quantile_forward_returns_with_membership(
        quantile_membership, window_labels, quantiles=quantiles
    )
    check_canceled()
    frames = {
        "coverage": coverage,
        "ic": ic,
        "book_returns": window_book_returns(book, window_labels),
        "tail_returns": window_tail_returns(tail, window_labels),
        "quantile_forward_returns": quantile_returns,
        "quantile_structure": quantile_curve_structure(
            quantile_returns, quantiles=quantiles
        ),
        "factor_returns": _window_factor_returns_prepared(
            prepared_factor,
            window_labels,
            complete=factor_standardization == "cross_sectional_zscore",
        ),
    }
    check_canceled()
    availability = window_labels.select(*_GROUP_COLUMNS, "available_date").unique()
    for family, frame in frames.items():
        if family != "coverage":
            frames[family] = frame.join(
                availability,
                on=list(_GROUP_COLUMNS),
                how="left",
                maintain_order="left",
            )
        if family in previous:
            refreshed = factor.select("evaluation_date").unique()
            retained = previous[family].join(
                refreshed, on="evaluation_date", how="anti"
            )
            frames[family] = (
                retained
                if frames[family].is_empty()
                else frames[family]
                if retained.is_empty()
                else pl.concat([retained, frames[family]], how="diagonal_relaxed")
            ).sort("evaluation_date")
    check_canceled()
    return frames


def _alpha_window_execution(
    options: BTExecutionOptions | None,
    windows: Sequence[SessionWindow],
    factor: pl.DataFrame,
    labels: pl.DataFrame,
    sessions: pl.DataFrame,
    book: pl.DataFrame,
    tail: pl.DataFrame,
) -> tuple[BTExecutionOptions, dict[str, int]]:
    """Reduce caller workers using conservative workspace estimates only."""
    selected = options or BTExecutionOptions()
    budget = selected.limits.memory_target_mib * 1024 * 1024
    shared = sum(
        frame.estimated_size() for frame in (factor, labels, sessions, book, tail)
    )
    reserved = budget // 5
    per_window = max(
        64 * 1024 * 1024, 8 * (factor.estimated_size() + labels.estimated_size())
    )
    memory_workers = max(1, (budget - reserved - shared) // per_window)
    workers = min(selected.workers, len(windows), memory_workers)
    # Very small panels do not amortize pool admission and per-window allocation.
    if factor.height < 1024:
        workers = 1
    return replace(selected, workers=workers), {
        "requested_workers": selected.workers,
        "actual_workers": workers,
        "shared_bytes": shared,
        "per_window_bytes": per_window,
        "reserved_bytes": reserved,
        "memory_budget_bytes": budget,
    }


def alpha_input_start(
    previous: EvaluationResult | None,
    *,
    calendar: pl.DataFrame,
    components: Sequence[str] | None = None,
    windows: Sequence[SessionWindow] = DAILY_SESSION_WINDOWS,
    persistence_lags: Sequence[int] = DAILY_SUMMARY_AUTOCORRELATION_LAGS,
    lead_lags: Sequence[int] = DAILY_BOOK_LEAD_LAGS,
    alpha_return_lags: Sequence[int] = DAILY_ALPHA_RETURN_LAGS,
) -> date:
    """Plan the earliest consumed Alpha/label coordinate from saved BT state.

    The calendar stays complete. Callers may read values and aligned labels
    only from this date, while supplying complete owner prefix evidence to
    ``evaluate_alpha``. New families initialize their required history; existing
    families restore checkpoints and preserve mature observations.
    """
    sessions = session_calendar(calendar, default=calendar["time"])
    days = sessions["time"].to_list()
    if not days:
        raise InputValidationError("Alpha input planning needs a nonempty calendar")
    if previous is None:
        return days[0]
    selected = set(_ALPHA_COMPONENTS if components is None else components)
    prior = set(previous.state.get("components", ()))
    if not prior or not prior <= selected or previous.state.get("through") is None:
        raise InputValidationError(
            "Alpha input planning needs complete saved component state"
        )
    new_paths = (
        selected & {"book_tail", "turnover"} and not prior & {"book_tail", "turnover"}
    ) or (selected - prior) & {"lead_lag", "alpha_return", "quantiles"}
    new_horizons = selected & {"horizons", "rolling_ic"} and not prior & {
        "horizons",
        "rolling_ic",
    }
    new_persistence = selected & {"horizons", "persistence"} and not prior & {
        "horizons",
        "persistence",
    }
    if new_paths or new_horizons or new_persistence:
        return days[0]
    try:
        position = days.index(date.fromisoformat(previous.state["through"]))
        history = max(
            [
                0,
                *(
                    window.end_session
                    for window in windows
                    if selected & {"horizons", "rolling_ic"}
                ),
                *(persistence_lags if selected & {"horizons", "persistence"} else ()),
                *(abs(lag) for lag in lead_lags if "lead_lag" in selected),
                *(alpha_return_lags if "alpha_return" in selected else ()),
            ]
        )
        first = days[max(0, position - history)]
        for name, checkpoint in previous.state.get("paths", {}).items():
            if checkpoint is None:
                # No return interval was processed for this path. Its original
                # decisions may mature long after the observation boundary.
                return days[0]
            position = days.index(date.fromisoformat(checkpoint["time"]))
            lag = (
                max([0, *lead_lags])
                if name.startswith("book_leadlag_")
                else max([0, *alpha_return_lags])
                if name.startswith("alpha_")
                else 0
            )
            first = min(first, days[max(0, position - lag)])
        coverage = previous.tables.get("horizon_coverage")
        if coverage is not None:
            pending = coverage.filter(~pl.col("is_mature"))["evaluation_date"].min()
            if pending is not None:
                first = min(first, pending)
        pending = previous.state.get("pending_horizon_start")
        if pending is not None:
            first = min(first, date.fromisoformat(pending))
        return first
    except (ValueError, KeyError, TypeError) as error:
        raise InputValidationError(
            "Alpha input planning needs saved boundaries in the complete calendar"
        ) from error


def alpha_update_plan(
    previous: EvaluationResult | Mapping[str, Any] | None,
    *,
    calendar: pl.DataFrame,
    available_date: date,
    affected_intervals: Sequence[tuple[date, date]] = (),
    preceding: Sequence[EvaluationResult | Mapping[str, Any]] = (),
    components: Sequence[str] | None = None,
    windows: Sequence[SessionWindow] = DAILY_SESSION_WINDOWS,
    persistence_lags: Sequence[int] = DAILY_SUMMARY_AUTOCORRELATION_LAGS,
    lead_lags: Sequence[int] = DAILY_BOOK_LEAD_LAGS,
    alpha_return_lags: Sequence[int] = DAILY_ALPHA_RETURN_LAGS,
    input_proof: Callable[[date, date, date | None], str | None] | None = None,
) -> dict[str, Any]:
    """Plan Alpha append/repair from saved metadata and source change intervals.

    ``previous_index`` is -1 for the current receipt, an index into ``preceding``
    for a proven earlier receipt, or None for initialization/blocked work. State
    mappings allow planning before historical primitive tables are read. The
    caller restores the selected immutable result and executes the existing
    ``evaluate_alpha(previous=...)`` path. Unknown impact or absent safe state
    never admits silent historical replay. Optional ``input_proof`` must use
    registered source-owner metadata; evaluation checks it again on consumption.
    """
    from bisect import bisect_left

    sessions = session_calendar(calendar, default=calendar["time"])
    days = sessions["time"].to_list()
    if not days:
        raise InputValidationError("Alpha update planning needs a nonempty calendar")
    if not isinstance(available_date, date):
        raise InputValidationError("Alpha update planning needs a date cutoff")
    options = dict(calendar=sessions, components=components, windows=windows,
                   persistence_lags=persistence_lags, lead_lags=lead_lags,
                   alpha_return_lags=alpha_return_lags)

    def result(status, *, index=None, start=None, output=None, reason=None):
        return {"status": status, "previous_index": index, "read_start": start,
                "output_start": output, "through": days[-1], "reason": reason}

    def saved(value):
        if (isinstance(value, Mapping)
                and set(value.get("components", ())) & {"horizons", "rolling_ic"}
                and "pending_horizon_start" not in value):
            raise ValueError(
                "saved Alpha metadata lacks pending-label scope; "
                "explicit state recovery is required"
            )
        return value if isinstance(value, EvaluationResult) else EvaluationResult(
            {}, {}, state=dict(value))

    def prefix_equal(candidate):
        state = candidate.state
        if state.get("input_prefix") is None or state.get("cutoff") is None:
            return False
        return input_proof is None or input_proof(
            days[0], date.fromisoformat(state["through"]),
            date.fromisoformat(state["cutoff"]),
        ) == state["input_prefix"]

    if previous is None:
        return result("initialize", start=days[0], output=days[0])
    try:
        current = saved(previous)
        boundary = date.fromisoformat(current.state["through"])
        cutoff = date.fromisoformat(current.state["cutoff"])
        if boundary not in days or available_date < cutoff:
            return result("blocked", reason=(
                "saved observation/information boundaries regress "
                "or leave the admitted calendar"))
        changes = []
        for first, last in affected_intervals:
            if (not isinstance(first, date) or not isinstance(last, date)
                    or first > last):
                raise InputValidationError(
                    "affected intervals require ordered date bounds")
            if first <= days[-1] and last >= days[0]:
                changes.append(max(first, days[0]))
        if not changes:
            if not prefix_equal(current):
                return result("blocked", reason=(
                    "changed input prefix requires source-owned "
                    "affected intervals and preceding state"))
            start = alpha_input_start(current, **options)
            selected = set(_ALPHA_COMPONENTS if components is None else components)
            if (days[-1] == boundary and available_date == cutoff
                    and selected == set(current.state.get("components", ()))):
                return result("up_to_date", index=-1)
            output = days[min(days.index(boundary) + 1, len(days) - 1)]
            return result("append", index=-1, start=start, output=output)
        selected = set(_ALPHA_COMPONENTS if components is None else components)
        halo = max([0,
            *(window.end_session for window in windows
              if selected & {"horizons", "rolling_ic"}),
            *(persistence_lags if selected & {"horizons", "persistence"} else ()),
            *(abs(lag) for lag in lead_lags if "lead_lag" in selected),
            *(alpha_return_lags if "alpha_return" in selected else ())])
        affected = days[max(0, bisect_left(days, min(changes)) - halo)]
        candidates = [(-1, current),
                      *((index, saved(value))
                        for index, value in enumerate(preceding))]
        candidates.sort(key=lambda pair: pair[1].state.get("through", ""), reverse=True)
        for index, candidate in candidates:
            state = candidate.state
            if (state.get("policy") != current.state.get("policy")
                    or set(state.get("components", ()))
                    != set(current.state.get("components", ()))):
                continue
            end = date.fromisoformat(state["through"])
            if (end >= affected or end not in days
                    or date.fromisoformat(state["cutoff"]) > available_date
                    or not prefix_equal(candidate)):
                continue
            if any(value is not None and date.fromisoformat(value["time"]) >= affected
                   for value in state.get("paths", {}).values()):
                continue
            start = alpha_input_start(candidate, **options)
            return result("repair", index=index, start=start,
                          output=days[days.index(end) + 1])
        return result("blocked", output=affected,
                      reason=("affected Alpha interval has no proven preceding "
                              "saved checkpoint; retain existing results and "
                              "restore original state before repair"))
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, InputValidationError):
            raise
        return result("blocked", reason=(
            "saved Alpha metadata needs explicit state recovery before planning"))


def evaluate_alpha(
    alpha: Node,
    forward_returns: pl.DataFrame,
    *,
    calendar: pl.DataFrame | None = None,
    available_date: date | None = None,
    components: Sequence[str] | None = None,
    windows: Sequence[SessionWindow] = DAILY_SESSION_WINDOWS,
    persistence_lags: Sequence[int] = DAILY_SUMMARY_AUTOCORRELATION_LAGS,
    annualization: int = 252,
    quantiles: int = 10,
    cost_rate: float = 0.0005,
    lead_lags: Sequence[int] = DAILY_BOOK_LEAD_LAGS,
    alpha_return_lags: Sequence[int] = DAILY_ALPHA_RETURN_LAGS,
    rolling_observations: int = 240,
    factor_standardization: str = "cross_sectional_zscore",
    options: BTExecutionOptions | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    check_canceled: Callable[[], None] | None = None,
    previous: EvaluationResult | None = None,
    input_proof: Callable[[date, date, date | None], str | None] | None = None,
    previous_input_proof: Callable[[date, date, date | None], str | None] | None = None,
) -> EvaluationResult:
    """Evaluate saved numeric/Prediction values without upstream production.

    Returns are already labeled by the caller. Their ``time`` is used exactly,
    including for cumulative/session-bucket labels; BT never inserts an
    execution lag or reconstructs returns from market prices. Requested
    diagnostics reuse the existing financial statistics and rank primitives.
    ``options`` bounds independent horizon windows; omitted options stay serial.
    ``progress`` reports completed/total windows on the coordinating thread.
    ``execution`` evidence is separate from numerical metrics and identity.
    ``previous`` reuses mature primitives and independent path checkpoints.
    ``input_proof(first, observation_through, information_cutoff)`` must bind
    the complete typed Alpha coordinates/Domain/traces and all caller labels
    consumed at that information boundary, including their economic intervals,
    availability and calendar mappings. Observation and information boundaries
    are independent: a later cutoff may consume labels beyond the saved Alpha
    boundary. Return ``None`` for unknown proof; never silently replay history.
    """

    allowed = _ALPHA_COMPONENTS
    selected = set(allowed if components is None else components)
    if not selected or selected - allowed:
        raise InputValidationError(
            f"unknown or empty alpha components: {sorted(selected - allowed)}"
        )
    return_statistics(pl.DataFrame(), annualization=annualization)
    if not math.isfinite(cost_rate) or cost_rate < 0:
        raise InputValidationError("cost_rate must be finite and nonnegative")
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or quantiles < 2:
        raise InputValidationError("quantiles must be an integer of at least two")
    cancel = check_canceled or (lambda: None)
    cancel()
    values = node_values(alpha, kind="alpha")
    sessions = session_calendar(calendar, default=alpha.domain.times)
    dates: list[date] = sessions["time"].to_list()
    if not dates:
        raise InputValidationError("Alpha evaluation needs a nonempty calendar")
    labels = normalize_forward_returns(
        forward_returns, calendar=sessions, available_date=available_date
    )
    if values.select("time").unique().join(sessions, on="time", how="anti").height:
        raise InputValidationError("alpha dates are absent from calendar")
    policy = hash_mapping(
        {
            "windows": [asdict(window) for window in windows],
            "persistence_lags": list(persistence_lags),
            "annualization": annualization,
            "quantiles": quantiles,
            "cost_rate": cost_rate,
            "lead_lags": list(lead_lags),
            "alpha_return_lags": list(alpha_return_lags),
            "rolling_observations": rolling_observations,
            "factor_standardization": factor_standardization,
        }
    )
    previous_state = {} if previous is None else previous.state
    prior_components = set(previous_state.get("components", ()))
    if previous is not None:
        if (
            previous_state.get("policy") != policy
            or input_proof is None
            or previous_state.get("input_prefix") is None
            or not prior_components
            or not prior_components <= selected
        ):
            raise InputValidationError(
                "Incremental Alpha evaluation needs saved policy/state and owner "
                "input-prefix evidence; explicit maintenance is required"
            )
        prior_end = date.fromisoformat(previous_state["through"])
        if (
            available_date is None
            or previous_state.get("cutoff") is None
            or available_date < date.fromisoformat(previous_state["cutoff"])
            or dates[-1] < prior_end
        ):
            raise InputValidationError(
                "Incremental Alpha evaluation needs an explicit "
                "nonregressing information cutoff and observation boundary"
            )
        if (previous_input_proof or input_proof)(
            dates[0],
            prior_end,
            date.fromisoformat(previous_state["cutoff"]),
        ) != previous_state["input_prefix"]:
            raise InputValidationError(
                "Alpha input prefix changed; plan affected intervals and restore "
                "a preceding saved checkpoint"
            )
        required_tables = set()
        if prior_components & {"horizons", "persistence"}:
            required_tables.update(
                (
                    "daily_signal_autocorrelation",
                    "horizon_signal_persistence",
                    "horizon_signal_persistence_summary",
                )
            )
        if prior_components & {"horizons", "rolling_ic"}:
            required_tables.update(
                f"horizon_{name}"
                for name in (
                    "coverage",
                    "ic",
                    "book_returns",
                    "tail_returns",
                    "quantile_forward_returns",
                    "quantile_structure",
                    "factor_returns",
                    "ic_summary",
                    "statistical_inference",
                )
            )
        if "rolling_ic" in prior_components:
            required_tables.add("daily_rolling_ic")
        if "book_tail" in prior_components:
            required_tables.update(("daily_book_returns", "daily_tail_returns"))
        for component, table in (
            ("turnover", "daily_book_turnover"),
            ("lead_lag", "daily_book_lead_lag_returns"),
            ("alpha_return", "daily_alpha_return_lag_returns"),
            ("quantiles", "daily_quantile_returns"),
        ):
            if component in prior_components:
                required_tables.add(table)
        if required_tables - set(previous.tables):
            raise InputValidationError(
                "Saved Alpha primitive families are incomplete; "
                "explicit primitive maintenance is required"
            )
        required_paths = set()
        if prior_components & {"book_tail", "turnover"}:
            required_paths.update(("book", "tail"))
        if "lead_lag" in prior_components:
            required_paths.update(f"book_leadlag_{lag}" for lag in lead_lags)
        if "alpha_return" in prior_components:
            required_paths.update(
                f"alpha_{kind}_{lag}"
                for kind in ("book", "tail")
                for lag in alpha_return_lags
            )
        if "quantiles" in prior_components:
            required_paths.update(
                f"quantile_{number}" for number in range(1, quantiles + 1)
            )
        if set(previous_state.get("paths", {})) != required_paths:
            raise InputValidationError(
                "Saved Alpha path checkpoints are incomplete; "
                "explicit state maintenance is required"
            )
        for name, checkpoint in previous_state.get("paths", {}).items():
            if any(
                f"state_{name}_{family}".replace("-", "minus") not in previous.tables
                for family in ("returns", "turnover", "coverage")
            ):
                raise InputValidationError(
                    "Saved Alpha path tables are incomplete; "
                    "explicit state maintenance is required"
                )
            returns = previous.tables[f"state_{name}_returns".replace("-", "minus")]
            if checkpoint is None and not returns.is_empty():
                raise InputValidationError(
                    "Saved Alpha path checkpoint is missing; "
                    "explicit state maintenance is required"
                )
        if (
            dates[-1] == prior_end
            and str(available_date) == previous_state["cutoff"]
            and selected == prior_components
        ):
            return replace(
                previous,
                execution={
                    "path_sessions": 0,
                    "continued_paths": 0,
                    "outcome": "up_to_date",
                },
            )
    cancel()
    factor = values.select(
        pl.col("time").alias("evaluation_date"),
        pl.col("time").alias("execution_date"),
        "asset_id",
        pl.col("value").alias("factor"),
    )
    tables: dict[str, pl.DataFrame] = {}
    execution: dict[str, Any] = {}
    state = {
        "policy": policy,
        "components": sorted(selected),
        "through": str(dates[-1]),
        "paths": {},
    }
    state["cutoff"] = None if available_date is None else str(available_date)
    state["input_prefix"] = (
        None
        if input_proof is None
        else input_proof(dates[0], dates[-1], available_date)
    )
    path_sessions = []

    def run_path(name, weights, rate, *, end=None):
        saved = previous_state.get("paths", {}).get(name)
        checkpoint = None
        if saved is not None:
            checkpoint = WeightStateCheckpoint(
                time=date.fromisoformat(saved["time"]),
                interval_end=date.fromisoformat(saved["interval_end"]),
                available_date=date.fromisoformat(saved["available_date"]),
                gross_value=saved["gross_value"],
                net_value=saved["net_value"],
                weights=pl.DataFrame(
                    saved["weights"],
                    schema={
                        "asset_id": pl.String,
                        "gross_weight": pl.Float64,
                        "net_weight": pl.Float64,
                    },
                ),
            )
        selected_labels = (
            labels
            if checkpoint is None
            else labels.filter(pl.col("time") > checkpoint.time)
        )
        selected_weights = (
            weights
            if checkpoint is None
            else weights.filter(pl.col("time") > checkpoint.time)
        )
        if end is not None:
            selected_labels = selected_labels.filter(pl.col("time") <= end)
        output, current = simulate_weight_returns(
            selected_weights,
            selected_labels,
            decisions=None,
            cost_rate=rate,
            checkpoint=checkpoint,
            include_holdings=False,
            check_canceled=cancel,
        )
        path_sessions.append(output["returns"].height)
        for family in ("returns", "turnover", "coverage"):
            table = f"state_{name}_{family}".replace("-", "minus")
            prior = None if previous is None else previous.tables.get(table)
            if prior is not None:
                if checkpoint is None and not prior.is_empty():
                    raise InputValidationError(
                        "Completed weight path lacks a continuation checkpoint; "
                        "explicit state maintenance is required"
                    )
                output[family] = pl.concat([prior, output[family]]).sort("time")
            tables[table] = output[family]
        state["paths"][name] = (
            None
            if current is None
            else {
                "time": str(current.time),
                "interval_end": str(current.interval_end),
                "available_date": str(current.available_date),
                "gross_value": current.gross_value,
                "net_value": current.net_value,
                "weights": current.weights.to_dict(as_series=False),
            }
        )
        return output

    cancel()
    metrics: dict[str, Any] = {"annualization": annualization, "quantiles": quantiles}
    if selected & {"horizons", "persistence"}:
        lags = _lags(persistence_lags)
        if min(lags) < 1:
            raise InputValidationError("persistence lags must be positive")
        persistence, persistence_summary = signal_rank_persistence(
            factor,
            calendar=sessions,
            horizons=lags,
            target_start=None
            if previous is None or not prior_components & {"horizons", "persistence"}
            else date.fromisoformat(previous_state["through"]),
        )
        if previous is not None and prior_components & {"horizons", "persistence"}:
            from .horizon import summarize_signal_persistence

            persistence = pl.concat(
                [previous.tables["daily_signal_autocorrelation"], persistence]
            ).sort("evaluation_date", "horizon_sessions")
            persistence_summary = summarize_signal_persistence(persistence, lags)
        tables["daily_signal_autocorrelation"] = persistence
        tables["horizon_signal_persistence"] = persistence
        tables["horizon_signal_persistence_summary"] = persistence_summary
    cancel()
    book = tail = membership = None
    work_factor = factor
    if previous is not None:
        work_start = alpha_input_start(
            previous,
            calendar=sessions,
            components=sorted(selected),
            windows=windows,
            persistence_lags=persistence_lags,
            lead_lags=lead_lags,
            alpha_return_lags=alpha_return_lags,
        )
        work_factor = factor.filter(pl.col("evaluation_date") >= work_start)
    if selected - {"persistence"}:
        normalized_factor = _validate_scheduled_factor_frame(work_factor)
        book = _centered_rank_book_weights_prepared(normalized_factor)
        cancel()
        membership = _quantile_membership(normalized_factor, quantiles=quantiles)
        tail = _gross_one_tail_weights_with_membership(membership, quantiles=quantiles)
    if selected & {"horizons", "rolling_ic"}:
        resolved_windows = tuple(windows)
        if not resolved_windows or any(
            not isinstance(window, SessionWindow) for window in resolved_windows
        ):
            raise InputValidationError("windows requires nonempty SessionWindow values")
        if len(
            {(window.window_kind, window.window_id) for window in resolved_windows}
        ) != len(resolved_windows):
            raise InputValidationError("window identities must be unique")
        assert book is not None and tail is not None and membership is not None
        prepared_factor = _prepare_factor_returns(work_factor, factor_standardization)
        window_options, execution = _alpha_window_execution(
            options, resolved_windows, factor, labels, sessions, book, tail
        )
        if progress:
            progress(0, len(resolved_windows))
        window_frames = run_evaluation_batch(
            resolved_windows,
            lambda window: _alpha_window_frames(
                window,
                factor=work_factor,
                labels=labels,
                sessions=sessions,
                book=book,
                tail=tail,
                quantiles=quantiles,
                factor_standardization=factor_standardization,
                quantile_membership=membership,
                prepared_factor=prepared_factor,
                check_canceled=cancel,
                previous=None
                if previous is None or not prior_components & {"horizons", "rolling_ic"}
                else {
                    name.removeprefix("horizon_"): frame.filter(
                        (pl.col("window_kind") == window.window_kind)
                        & (pl.col("window_id") == window.window_id)
                    )
                    for name, frame in previous.tables.items()
                    if name
                    in {
                        "horizon_coverage",
                        "horizon_ic",
                        "horizon_book_returns",
                        "horizon_tail_returns",
                        "horizon_quantile_forward_returns",
                        "horizon_quantile_structure",
                        "horizon_factor_returns",
                    }
                },
            ),
            options=window_options,
            check_canceled=cancel,
            progress=progress or (lambda *_: None),
        )
        cancel()
        families = {
            name: [frames[name] for frames in window_frames]
            for name in window_frames[0]
        }
        for name, frames in families.items():
            tables[f"horizon_{name}"] = pl.concat(frames, how="diagonal_relaxed")
            if "execution_date" in tables[f"horizon_{name}"].columns:
                tables[f"horizon_{name}"] = tables[f"horizon_{name}"].with_columns(
                    pl.col("execution_date").alias("target_start_date")
                )
        cancel()
        tables["horizon_ic_summary"] = summarize_window_ic(
            tables["horizon_ic"], annualization_sessions=annualization
        )
        window_availability = (
            tables["horizon_ic"]
            .group_by(*_WINDOW_COLUMNS)
            .agg(pl.col("available_date").max())
        )
        tables["horizon_ic_summary"] = tables["horizon_ic_summary"].join(
            window_availability,
            on=list(_WINDOW_COLUMNS),
            how="left",
            maintain_order="left",
        )
        cancel()
        tables["horizon_statistical_inference"] = build_statistical_inference(
            ic=tables["horizon_ic"],
            book_returns=tables["horizon_book_returns"],
            tail_returns=tables["horizon_tail_returns"],
            quantile_structure=tables["horizon_quantile_structure"],
            factor_returns=tables["horizon_factor_returns"],
        )
        tables["horizon_statistical_inference"] = tables[
            "horizon_statistical_inference"
        ].join(
            window_availability,
            on=list(_WINDOW_COLUMNS),
            how="left",
            maintain_order="left",
        )
        metrics["ic"] = tables["horizon_ic_summary"].to_dicts()
        if "rolling_ic" in selected:
            cancel()
            tables["daily_rolling_ic"] = rolling_window_information_coefficients(
                tables["horizon_ic"],
                observations=rolling_observations,
                previous=None
                if previous is None or "rolling_ic" not in prior_components
                else previous.tables["daily_rolling_ic"].drop("available_date"),
                previous_observations=rolling_observations,
            )
            rolling_availability = (
                tables["horizon_ic"]
                .sort("evaluation_date")
                .with_columns(pl.col("available_date").cum_max().over(*_WINDOW_COLUMNS))
                .select("evaluation_date", *_WINDOW_COLUMNS, "available_date")
            )
            tables["daily_rolling_ic"] = tables["daily_rolling_ic"].join(
                rolling_availability,
                on=["evaluation_date", *_WINDOW_COLUMNS],
                how="left",
                maintain_order="left",
            )
    cancel()
    if selected & {"book_tail", "turnover", "lead_lag", "alpha_return", "quantiles"}:
        assert book is not None and tail is not None
        book_values = book.filter(pl.col("book_weight").is_not_null()).select(
            pl.col("evaluation_date").alias("time"),
            "asset_id",
            pl.col("book_weight").alias("value"),
        )
        tail_values = tail.filter(pl.col("tail_weight").is_not_null()).select(
            pl.col("evaluation_date").alias("time"),
            "asset_id",
            pl.col("tail_weight").alias("value"),
        )
        if selected & {"book_tail", "turnover"}:
            for path_kind, weights in (("book", book_values), ("tail", tail_values)):
                output = run_path(path_kind, weights, cost_rate)
                if "book_tail" in selected:
                    tables[f"daily_{path_kind}_returns"] = output["returns"]
                    metrics[path_kind] = return_statistics(
                        output["returns"], annualization=annualization
                    )
                if "turnover" in selected and path_kind == "book":
                    tables["daily_book_turnover"] = output["turnover"]
        if "lead_lag" in selected:
            tables["daily_book_lead_lag_returns"] = _path_frames(
                book_values,
                labels,
                sessions,
                _lags(lead_lags, negative=True),
                cost_rate,
                check_canceled,
                path_runner=run_path,
                path_id="book_leadlag",
                previous=None
                if previous is None
                else previous.tables.get("daily_book_lead_lag_returns"),
            )
        if "alpha_return" in selected:
            frames = [
                _path_frames(
                    weights,
                    labels,
                    sessions,
                    _lags(alpha_return_lags),
                    cost_rate,
                    check_canceled,
                    path_runner=run_path,
                    path_id=f"alpha_{kind}",
                    previous=None
                    if previous is None or "alpha_return" not in prior_components
                    else previous.tables.get(
                        "daily_alpha_return_lag_returns", pl.DataFrame()
                    )
                    .filter(pl.col("path_kind") == kind)
                    .drop("path_kind"),
                ).with_columns(pl.lit(kind).alias("path_kind"))
                for kind, weights in (("book", book_values), ("tail", tail_values))
            ]
            common = _common_lag_dates(frames)
            tables["daily_alpha_return_lag_returns"] = pl.concat(frames).join(
                common, on="time", how="inner"
            )
        if "quantiles" in selected:
            assert membership is not None
            valid = membership.filter(
                (pl.col("_count") >= quantiles) & (pl.col("_unique") >= 2)
            )
            frames = []
            for number in range(1, quantiles + 1):
                weights = valid.filter(pl.col("quantile") == f"q{number}").with_columns(
                    (1.0 / pl.len().over("evaluation_date")).alias("value")
                )
                output = run_path(
                    f"quantile_{number}",
                    weights.select(
                        pl.col("evaluation_date").alias("time"), "asset_id", "value"
                    ),
                    0.0,
                )
                frames.append(
                    output["returns"]
                    .join(
                        output["coverage"].select(
                            "time", pl.col("expected_count").alias("constituent_count")
                        ),
                        on="time",
                        how="left",
                    )
                    .with_columns(
                        pl.lit(f"q{number}").alias("quantile"),
                        pl.lit(None, dtype=pl.String).alias("unavailable_reason"),
                    )
                )
            tables["daily_quantile_returns"] = pl.concat(frames)
    cancel()
    execution["path_sessions"] = sum(path_sessions)
    execution["continued_paths"] = sum(
        item is not None for item in previous_state.get("paths", {}).values()
    )
    coverage = tables.get("horizon_coverage")
    pending = (None if coverage is None else
               coverage.filter(~pl.col("is_mature"))["evaluation_date"].min())
    state["pending_horizon_start"] = None if pending is None else str(pending)
    return EvaluationResult(tables, metrics, execution=execution, state=state)


__all__ = [
    "EvaluationResult",
    "alpha_input_start",
    "evaluate_alpha",
    "evaluate_weights",
    "return_statistics",
]
