"""Standalone evaluation of saved Alpha values and Portfolio weights."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl
from bagelquant_core import Node

from .exceptions import InputValidationError
from .horizon import (
    DAILY_ALPHA_RETURN_LAGS,
    DAILY_BOOK_LEAD_LAGS,
    DAILY_SESSION_WINDOWS,
    DAILY_SUMMARY_AUTOCORRELATION_LAGS,
    SessionWindow,
    _quantile_membership,
    build_statistical_inference,
    centered_rank_book_weights,
    gross_one_tail_weights,
    quantile_curve_structure,
    rolling_window_information_coefficients,
    signal_rank_persistence,
    summarize_window_ic,
    window_book_returns,
    window_factor_returns,
    window_information_coefficients,
    window_quantile_forward_returns,
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
from .statistics import return_statistics


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Numerical tables and metrics, with an optional resumable weight state."""

    tables: dict[str, pl.DataFrame]
    metrics: dict[str, Any]
    checkpoint: WeightStateCheckpoint | None = None


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
) -> pl.DataFrame:
    frames = []
    for lag in lags:
        if check_canceled:
            check_canceled()
        shifted = _shift_weights(weights, calendar, lag)
        tables, _ = simulate_weight_returns(
            shifted,
            labels,
            decisions=None,
            cost_rate=cost_rate,
            include_holdings=False,
            check_canceled=check_canceled,
        )
        frames.append(
            tables["returns"]
            .join(shifted.select("time").unique(), on="time", how="inner")
            .with_columns(pl.lit(lag, dtype=pl.Int64).alias("lag"))
        )
    return (
        pl.concat(frames)
        .join(_common_lag_dates(frames), on="time", how="inner")
        .sort("lag", "time")
    )


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
    check_canceled: Callable[[], None] | None = None,
) -> EvaluationResult:
    """Evaluate saved numeric/Prediction values without upstream production.

    Returns are already labeled by the caller. Their ``time`` is used exactly,
    including for cumulative/session-bucket labels; BT never inserts an
    execution lag or reconstructs returns from market prices. Requested
    diagnostics reuse the existing financial statistics and rank primitives.
    """

    allowed = {
        "horizons",
        "rolling_ic",
        "book_tail",
        "turnover",
        "lead_lag",
        "persistence",
        "alpha_return",
        "quantiles",
    }
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
    values = node_values(alpha, kind="alpha")
    sessions = session_calendar(calendar, default=alpha.domain.times)
    labels = normalize_forward_returns(
        forward_returns, calendar=sessions, available_date=available_date
    )
    if values.select("time").unique().join(sessions, on="time", how="anti").height:
        raise InputValidationError("alpha dates are absent from calendar")
    factor = values.select(
        pl.col("time").alias("evaluation_date"),
        pl.col("time").alias("execution_date"),
        "asset_id",
        pl.col("value").alias("factor"),
    )
    tables: dict[str, pl.DataFrame] = {}
    metrics: dict[str, Any] = {"annualization": annualization, "quantiles": quantiles}
    if check_canceled:
        check_canceled()
    if selected & {"horizons", "persistence"}:
        lags = _lags(persistence_lags)
        if min(lags) < 1:
            raise InputValidationError("persistence lags must be positive")
        persistence, persistence_summary = signal_rank_persistence(
            factor, calendar=sessions, horizons=lags
        )
        tables["daily_signal_autocorrelation"] = persistence
        tables["horizon_signal_persistence"] = persistence
        tables["horizon_signal_persistence_summary"] = persistence_summary
    book = tail = None
    if selected - {"persistence"}:
        book = centered_rank_book_weights(factor)
        tail = gross_one_tail_weights(factor, quantiles=quantiles)
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
        families: dict[str, list[pl.DataFrame]] = {
            name: []
            for name in (
                "coverage",
                "ic",
                "book_returns",
                "tail_returns",
                "quantile_forward_returns",
                "quantile_structure",
                "factor_returns",
            )
        }
        for window in resolved_windows:
            if check_canceled:
                check_canceled()
            window_labels, coverage = _window_labels(factor, labels, sessions, window)
            ic = window_information_coefficients(factor, window_labels)
            quantile_returns = window_quantile_forward_returns(
                factor, window_labels, quantiles=quantiles
            )
            families["coverage"].append(coverage)
            families["ic"].append(ic)
            families["book_returns"].append(window_book_returns(book, window_labels))
            families["tail_returns"].append(window_tail_returns(tail, window_labels))
            families["quantile_forward_returns"].append(quantile_returns)
            families["quantile_structure"].append(
                quantile_curve_structure(quantile_returns, quantiles=quantiles)
            )
            families["factor_returns"].append(
                window_factor_returns(
                    factor, window_labels, standardization=factor_standardization
                )
            )
            availability = window_labels.select(
                *_GROUP_COLUMNS, "available_date"
            ).unique()
            for family in families:
                if family != "coverage":
                    families[family][-1] = families[family][-1].join(
                        availability,
                        on=list(_GROUP_COLUMNS),
                        how="left",
                        maintain_order="left",
                    )
        for name, frames in families.items():
            tables[f"horizon_{name}"] = pl.concat(frames, how="diagonal_relaxed")
            if "execution_date" in tables[f"horizon_{name}"].columns:
                tables[f"horizon_{name}"] = tables[f"horizon_{name}"].with_columns(
                    pl.col("execution_date").alias("target_start_date")
                )
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
            tables["daily_rolling_ic"] = rolling_window_information_coefficients(
                tables["horizon_ic"], observations=rolling_observations
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
                output, _ = simulate_weight_returns(
                    weights,
                    labels,
                    decisions=None,
                    cost_rate=cost_rate,
                    include_holdings=False,
                    check_canceled=check_canceled,
                )
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
                ).with_columns(pl.lit(kind).alias("path_kind"))
                for kind, weights in (("book", book_values), ("tail", tail_values))
            ]
            common = _common_lag_dates(frames)
            tables["daily_alpha_return_lag_returns"] = pl.concat(frames).join(
                common, on="time", how="inner"
            )
        if "quantiles" in selected:
            membership = _quantile_membership(factor, quantiles=quantiles)
            valid = membership.filter(
                (pl.col("_count") >= quantiles) & (pl.col("_unique") >= 2)
            )
            frames = []
            for number in range(1, quantiles + 1):
                weights = valid.filter(pl.col("quantile") == f"q{number}").with_columns(
                    (1.0 / pl.len().over("evaluation_date")).alias("value")
                )
                output, _ = simulate_weight_returns(
                    weights.select(
                        pl.col("evaluation_date").alias("time"), "asset_id", "value"
                    ),
                    labels,
                    decisions=None,
                    cost_rate=0.0,
                    include_holdings=False,
                    check_canceled=check_canceled,
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
    return EvaluationResult(tables, metrics)


__all__ = [
    "EvaluationResult",
    "evaluate_alpha",
    "evaluate_weights",
    "return_statistics",
]
