"""Multi-horizon cross-sectional prediction diagnostics.

The primitives in this module deliberately stop before portfolio simulation.
They evaluate one point-in-time prediction cross-section against explicit
session windows while preserving the evaluation-to-execution lineage carried
by :class:`~bagelquant_bt.policy.ScheduledPrediction`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats

from .config import BacktestConfig
from .engine import (
    _attach_slippage_rates,
    _sparse_executed_turnover,
)
from .inputs import (
    ASSET_ID,
    TIME,
    validate_execution_availability,
    validate_panel_frame,
    validate_prices,
    validate_slippage_rates,
)
from .policy import ScheduledPrediction
from .returns import _prepare_price_data

WindowKind = Literal["cumulative", "bucket"]
DailyDiagnosticProgress = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """One return window expressed in trading-session offsets.

    ``start_session`` and ``end_session`` are one-based return-session
    numbers.  The corresponding price offsets are therefore
    ``start_session - 1`` and ``end_session`` from the execution session.
    """

    window_kind: WindowKind
    window_id: str
    start_session: int
    end_session: int

    def __post_init__(self) -> None:
        if self.window_kind not in {"cumulative", "bucket"}:
            raise ValueError(f"unsupported window kind: {self.window_kind}")
        if not self.window_id.strip():
            raise ValueError("window_id must not be blank")
        if self.start_session < 1 or self.end_session < self.start_session:
            raise ValueError(
                "session windows require 1 <= start_session <= end_session"
            )
        if self.window_kind == "cumulative" and self.start_session != 1:
            raise ValueError("cumulative windows must start at session one")

    @property
    def start_offset(self) -> int:
        return self.start_session - 1

    @property
    def end_offset(self) -> int:
        return self.end_session

    @property
    def width(self) -> int:
        return self.end_session - self.start_session + 1


@dataclass(frozen=True, slots=True)
class HACMeanTest:
    """Bartlett Newey-West inference for a sample mean."""

    mean: float | None
    standard_error: float | None
    t_value: float | None
    p_value: float | None
    confidence_low: float | None
    confidence_high: float | None
    sample_size: int
    lag: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PredictionHorizonDiagnostics:
    """Aggregate prediction diagnostics for a fixed collection of windows.

    Row-level forward labels remain available from
    :func:`session_window_forward_returns`.  The complete diagnostic result
    deliberately retains only the aggregate outputs consumed by research so a
    caller never has to keep every asset/window label resident at once.
    """

    coverage: pl.DataFrame
    ic: pl.DataFrame
    ic_summary: pl.DataFrame
    book_returns: pl.DataFrame
    tail_returns: pl.DataFrame
    quantile_forward_returns: pl.DataFrame
    quantile_structure: pl.DataFrame
    factor_returns: pl.DataFrame
    signal_persistence: pl.DataFrame
    signal_persistence_summary: pl.DataFrame
    statistical_inference: pl.DataFrame
    max_window_forward_rows: int


@dataclass(frozen=True, slots=True)
class DailyRankPathDiagnostics:
    """Daily diagnostic rank paths and causal summary inputs.

    These paths are research diagnostics, not a production Weight Policy or
    simulated account.  Gross returns use requested rank weights.  Net returns
    subtract proportional transaction costs and slippage from requested
    turnover without capital, cash, minimum fees, or insolvency state.
    Execution constraints affect only the separately reported executed
    turnover diagnostic.
    """

    book_daily_returns: pl.DataFrame
    tail_daily_returns: pl.DataFrame
    quantile_returns: pl.DataFrame
    book_turnover: pl.DataFrame
    book_lead_lag_returns: pl.DataFrame
    alpha_return_lag_returns: pl.DataFrame
    signal_autocorrelation: pl.DataFrame
    rolling_ic: pl.DataFrame


@dataclass(frozen=True, slots=True)
class DailyPredictionDiagnostics:
    """Complete daily prediction diagnostics produced from one prepared input.

    The aggregate horizon result and the daily rank paths deliberately share
    validated Signal, market, calendar, and rank-weight preparation.  Asset
    level horizon labels remain window-scoped and are released before the
    daily paths are built.
    """

    horizons: PredictionHorizonDiagnostics
    paths: DailyRankPathDiagnostics


@dataclass(frozen=True, slots=True)
class _PreparedDailyDiagnostics:
    factor: pl.DataFrame
    market: pl.DataFrame
    calendar: pl.DataFrame
    sessions: tuple[date, ...]
    market_sessions: tuple[date, ...]
    book: pl.DataFrame | None = None
    tail: pl.DataFrame | None = None
    quantile_membership: pl.DataFrame | None = None
    price_lookup: pl.DataFrame | None = None


DAILY_CUMULATIVE_WINDOWS = tuple(
    SessionWindow("cumulative", f"cumulative_{horizon}d", 1, horizon)
    for horizon in (1, 5, 10, 20, 40, 60, 120)
)
DAILY_BUCKET_WINDOWS = (
    SessionWindow("bucket", "bucket_1d", 1, 1),
    SessionWindow("bucket", "bucket_2_5d", 2, 5),
    SessionWindow("bucket", "bucket_6_20d", 6, 20),
    SessionWindow("bucket", "bucket_21_60d", 21, 60),
    SessionWindow("bucket", "bucket_61_120d", 61, 120),
)
DAILY_SESSION_WINDOWS = DAILY_CUMULATIVE_WINDOWS + DAILY_BUCKET_WINDOWS
SIGNAL_PERSISTENCE_HORIZONS = (1, 5, 10, 20, 40, 60, 120)
DAILY_SUMMARY_AUTOCORRELATION_LAGS = tuple(range(1, 121))
DAILY_BOOK_LEAD_LAGS = tuple(range(-30, 31))
DAILY_ALPHA_RETURN_LAGS = (0, 1, 2, 5, 10, 20, 60)
DAILY_ROLLING_IC_OBSERVATIONS = 240
_PERSISTENCE_PAIR_ROW_BUDGET = 250_000
_DAILY_RANK_LINEAGE_ROW_BUDGET = 250_000
_DAILY_BOOK_QUANTILE_JOIN_ROW_BUDGET = 250_000

_WINDOW_COLUMNS = (
    "window_kind",
    "window_id",
    "start_session",
    "end_session",
)
_RETURN_GROUP_COLUMNS = (
    "evaluation_date",
    "execution_date",
    "target_end_date",
    *_WINDOW_COLUMNS,
)


def session_window_forward_returns(
    signals: ScheduledPrediction,
    prices: pl.DataFrame,
    *,
    windows: Sequence[SessionWindow] = DAILY_SESSION_WINDOWS,
    calendar: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build fixed-session forward returns without future cross-section changes.

    A position requires an exact, finite execution price.  Once executable,
    an asset-specific price gap freezes that position at its last observation;
    the cumulative move is recognized when pricing resumes.  Evaluation/window
    pairs whose target session has not occurred are omitted from the label
    frame and reported as incomplete in the companion coverage frame.
    """

    factor = _scheduled_factor(signals)
    market = validate_prices(prices)
    resolved_windows = _validate_windows(windows)
    sessions = _session_dates(calendar, market)
    return _session_window_forward_returns_from_frames(
        factor,
        market,
        windows=resolved_windows,
        sessions=sessions,
    )


def _session_window_forward_returns_from_frames(
    factor: pl.DataFrame,
    market: pl.DataFrame,
    *,
    windows: Sequence[SessionWindow],
    sessions: Sequence[date],
    price_lookup: pl.DataFrame | None = None,
    compact_window_keys: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build labels from validated shared inputs for one or more windows."""

    window_schedule = _window_schedule(
        factor.select("evaluation_date", "execution_date").unique(),
        sessions,
        windows,
    )
    expected = (
        factor.group_by("evaluation_date", "execution_date")
        .agg(pl.len().alias("expected_count"))
        .sort("evaluation_date")
    )
    coverage = (
        window_schedule.join(
            expected,
            on=["evaluation_date", "execution_date"],
            how="left",
        )
        .with_columns(pl.col("expected_count").fill_null(0).cast(pl.Int64))
        .select(
            "evaluation_date",
            "execution_date",
            "target_end_date",
            *_WINDOW_COLUMNS,
            "start_offset",
            "end_offset",
            "target_available",
            "expected_count",
        )
    )
    available = window_schedule.filter(pl.col("target_available"))
    if compact_window_keys:
        available = available.with_columns(
            pl.col("window_kind").cast(pl.Categorical),
            pl.col("window_id").cast(pl.Categorical),
            pl.col("start_session").cast(pl.UInt8),
            pl.col("end_session").cast(pl.UInt8),
            pl.col("start_offset").cast(pl.UInt8),
            pl.col("end_offset").cast(pl.UInt8),
        )
    if factor.is_empty() or available.is_empty():
        return _empty_forward_returns(), coverage.with_columns(
            pl.lit(0, dtype=pl.Int64).alias("observed_count"),
            pl.lit(None, dtype=pl.Float64).alias("coverage_ratio"),
        )

    rows = factor.join(
        available,
        on=["evaluation_date", "execution_date"],
        how="inner",
    ).with_row_index("_row_id")
    if "_execution_price" not in rows.columns:
        exact_execution = market.select(
            pl.col(TIME).alias("execution_date"),
            ASSET_ID,
            pl.col("price").alias("_execution_price"),
        )
        rows = rows.join(
            exact_execution,
            on=["execution_date", ASSET_ID],
            how="left",
        )
    resolved_price_lookup = (
        market.select(
            ASSET_ID,
            pl.col(TIME).alias("_observed_date"),
            "price",
        ).sort([ASSET_ID, "_observed_date"])
        if price_lookup is None
        else price_lookup
    )
    if all(window.start_offset == 0 for window in windows):
        rows = rows.with_columns(pl.col("_execution_price").alias("_start_price"))
    else:
        rows = _attach_asof_price(
            rows,
            resolved_price_lookup,
            lookup_column="_start_price_date",
            output_column="_start_price",
        ).with_columns(
            pl.when(pl.col("start_offset") == 0)
            .then(pl.col("_execution_price"))
            .otherwise(pl.col("_start_price"))
            .alias("_start_price")
        )
    rows = _attach_asof_price(
        rows,
        resolved_price_lookup,
        lookup_column="target_end_date",
        output_column="_end_price",
    )
    forward = (
        rows.with_columns(
            pl.when(
                pl.col("_execution_price").is_not_null()
                & pl.col("_start_price").is_not_null()
                & pl.col("_end_price").is_not_null()
            )
            .then(pl.col("_end_price") / pl.col("_start_price") - 1.0)
            .otherwise(None)
            .alias("forward_return")
        )
        .select(
            "evaluation_date",
            "execution_date",
            "target_end_date",
            ASSET_ID,
            "window_kind",
            "window_id",
            "start_session",
            "end_session",
            "start_offset",
            "end_offset",
            "forward_return",
            *(
                [
                    "_quantile_number",
                    "_quantile_count",
                    "_quantile_unique_valid",
                ]
                if "_quantile_number" in rows.columns
                else []
            ),
        )
        .sort(["evaluation_date", "window_id", ASSET_ID])
    )
    if compact_window_keys:
        forward = forward.drop("start_offset", "end_offset")
    observed = forward.group_by(*_RETURN_GROUP_COLUMNS).agg(
        pl.col("forward_return").is_not_null().sum().alias("observed_count")
    )
    if compact_window_keys:
        observed = _restore_window_key_dtypes(observed)
    coverage = (
        coverage.join(observed, on=list(_RETURN_GROUP_COLUMNS), how="left")
        .with_columns(pl.col("observed_count").fill_null(0).cast(pl.Int64))
        .with_columns(
            pl.when(pl.col("target_available") & (pl.col("expected_count") > 0))
            .then(pl.col("observed_count") / pl.col("expected_count"))
            .otherwise(None)
            .alias("coverage_ratio")
        )
        .sort(["evaluation_date", "window_id"])
    )
    return forward, coverage


def _restore_window_key_dtypes(frame: pl.DataFrame) -> pl.DataFrame:
    """Restore the stable public schema after compact single-window aggregation."""

    return frame.with_columns(
        pl.col("window_kind").cast(pl.String),
        pl.col("window_id").cast(pl.String),
        pl.col("start_session").cast(pl.Int64),
        pl.col("end_session").cast(pl.Int64),
    )


def centered_rank_book_weights(factor: pl.DataFrame) -> pl.DataFrame:
    """Create centered average-rank weights with net zero and gross one."""

    normalized = _validate_scheduled_factor_frame(factor)
    return _centered_rank_book_weights_prepared(normalized)


def _centered_rank_book_weights_prepared(
    normalized: pl.DataFrame,
) -> pl.DataFrame:
    """Build Book weights from a caller-validated factor frame."""

    if normalized.is_empty():
        return _empty_weight_frame("book_weight")
    return (
        _with_centered_rank_book_weights(normalized)
        .select(
            "evaluation_date",
            "execution_date",
            ASSET_ID,
            "book_weight",
            "unavailable_reason",
        )
        .sort(["evaluation_date", ASSET_ID])
    )


def _with_centered_rank_book_weights(normalized: pl.DataFrame) -> pl.DataFrame:
    """Attach Book weights before any deterministic quantile reordering."""

    ranked = normalized.with_columns(
        (pl.col("factor").rank("average") / pl.len())
        .over("evaluation_date")
        .alias("_percentile_rank"),
        pl.len().over("evaluation_date").alias("_count"),
        pl.col("factor").n_unique().over("evaluation_date").alias("_unique"),
    ).with_columns(
        (
            pl.col("_percentile_rank")
            - pl.col("_percentile_rank").mean().over("evaluation_date")
        ).alias("_centered")
    )
    ranked = ranked.with_columns(
        pl.col("_centered").abs().sum().over("evaluation_date").alias("_gross"),
        (pl.col("_centered") > 0).any().over("evaluation_date").alias("_has_long"),
        (pl.col("_centered") < 0).any().over("evaluation_date").alias("_has_short"),
    )
    valid = (
        (pl.col("_count") >= 2)
        & (pl.col("_unique") >= 2)
        & pl.col("_has_long")
        & pl.col("_has_short")
        & (pl.col("_gross") > 0)
    )
    return ranked.with_columns(
        pl.when(valid)
        .then(pl.col("_centered") / pl.col("_gross"))
        .otherwise(None)
        .alias("book_weight"),
        pl.when(valid)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.lit("book requires at least two non-constant ranks"))
        .alias("unavailable_reason"),
    ).drop("_percentile_rank", "_centered", "_gross", "_has_long", "_has_short")


def _centered_rank_book_active_lineage(normalized: pl.DataFrame) -> pl.Series:
    """Return the v5-order active-weight mask on the canonical key ordering."""

    runs = normalized.group_by("evaluation_date", maintain_order=True).len()
    frames: list[pl.Series] = []
    offset = 0
    chunk_offset = 0
    chunk_rows = 0
    for _session, row_count in runs.iter_rows():
        resolved_count = int(row_count)
        if chunk_rows and chunk_rows + resolved_count > _DAILY_RANK_LINEAGE_ROW_BUDGET:
            frames.append(
                _centered_rank_book_active_chunk(
                    normalized.slice(chunk_offset, chunk_rows)
                )
            )
            chunk_offset = offset
            chunk_rows = 0
        chunk_rows += resolved_count
        offset += resolved_count
    if chunk_rows:
        frames.append(
            _centered_rank_book_active_chunk(normalized.slice(chunk_offset, chunk_rows))
        )
    return pl.concat(frames)


def _centered_rank_book_active_chunk(normalized: pl.DataFrame) -> pl.Series:
    ranked = normalized.with_columns(
        (pl.col("factor").rank("average") / pl.len())
        .over("evaluation_date")
        .alias("_percentile_rank"),
    )
    return ranked.select(
        (
            pl.col("_percentile_rank")
            != pl.col("_percentile_rank").mean().over("evaluation_date")
        ).alias("_book_active")
    ).get_column("_book_active")


def gross_one_tail_weights(
    factor: pl.DataFrame,
    *,
    quantiles: int = 10,
) -> pl.DataFrame:
    """Create q1/qN tail weights with long/short books of one half each."""

    if quantiles < 2:
        raise ValueError("tail weights require at least two quantiles")
    normalized = _validate_scheduled_factor_frame(factor)
    if normalized.is_empty():
        return _empty_weight_frame("tail_weight")
    bucketed = _quantile_membership(normalized, quantiles=quantiles)
    return _gross_one_tail_weights_with_membership(
        bucketed,
        quantiles=quantiles,
    )


def _gross_one_tail_weights_with_membership(
    bucketed: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    """Build tail weights from one caller-reused quantile assignment."""

    valid = (
        (pl.col("_count") >= quantiles)
        & (pl.col("_unique") >= 2)
        & pl.col("quantile").is_not_null()
    )
    return (
        bucketed.with_columns(
            pl.when(valid & (pl.col("quantile") == "q1"))
            .then(0.5 / pl.len().over("evaluation_date", "quantile"))
            .when(valid & (pl.col("quantile") == f"q{quantiles}"))
            .then(-0.5 / pl.len().over("evaluation_date", "quantile"))
            .when(valid)
            .then(0.0)
            .otherwise(None)
            .alias("tail_weight"),
            pl.when(valid)
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(
                pl.lit(f"tail requires {quantiles} assets and a non-constant signal")
            )
            .alias("unavailable_reason"),
        )
        .select(
            "evaluation_date",
            "execution_date",
            ASSET_ID,
            "quantile",
            "tail_weight",
            "unavailable_reason",
        )
        .sort(["evaluation_date", ASSET_ID])
    )


def window_information_coefficients(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Compute Pearson and average-rank Spearman IC for every window."""

    normalized = _validate_scheduled_factor_frame(factor)
    paired = forward_returns.join(
        normalized.select("evaluation_date", ASSET_ID, "factor"),
        on=["evaluation_date", ASSET_ID],
        how="left",
    )
    if paired.is_empty():
        return _empty_window_metric(
            {"pearson_ic": pl.Float64, "spearman_ic": pl.Float64}
        )
    paired = paired.with_columns(
        pl.col("forward_return")
        .is_not_null()
        .sum()
        .over(*_RETURN_GROUP_COLUMNS)
        .cast(pl.Int64)
        .alias("sample_size"),
        pl.col("factor")
        .rank("average")
        .over(*_RETURN_GROUP_COLUMNS)
        .alias("_factor_rank"),
        pl.col("forward_return")
        .rank("average")
        .over(*_RETURN_GROUP_COLUMNS)
        .alias("_return_rank"),
    )
    values = paired.drop_nulls(["factor", "forward_return"])
    metrics = (
        values.group_by(*_RETURN_GROUP_COLUMNS)
        .agg(
            _safe_corr("factor", "forward_return").alias("pearson_ic"),
            _safe_corr("_factor_rank", "_return_rank").alias("spearman_ic"),
            pl.first("sample_size").alias("sample_size"),
        )
        .sort(["evaluation_date", "window_id"])
    )
    grid = paired.select(*_RETURN_GROUP_COLUMNS).unique()
    return (
        grid.join(metrics, on=list(_RETURN_GROUP_COLUMNS), how="left")
        .with_columns(pl.col("sample_size").fill_null(0).cast(pl.Int64))
        .sort(["evaluation_date", "window_id"])
    )


def window_book_returns(
    weights: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate centered-rank book returns without label-time reweighting."""

    return _weighted_window_returns(
        weights,
        forward_returns,
        weight_column="book_weight",
        return_column="book_return",
    )


def window_tail_returns(
    weights: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate gross-one q1/qN tail returns without future reweighting."""

    return _weighted_window_returns(
        weights,
        forward_returns,
        weight_column="tail_weight",
        return_column="tail_return",
    )


def window_quantile_forward_returns(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    quantiles: int = 10,
) -> pl.DataFrame:
    """Return complete q1-to-qN mean-forward-return curves per window."""

    membership = _quantile_membership(
        _validate_scheduled_factor_frame(factor), quantiles=quantiles
    )
    return _window_quantile_forward_returns_with_membership(
        membership,
        forward_returns,
        quantiles=quantiles,
    )


def _window_quantile_forward_returns_with_membership(
    membership: pl.DataFrame | None,
    forward_returns: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    """Aggregate one label window with a caller-reused quantile membership."""

    if forward_returns.is_empty():
        return _empty_window_metric(
            {
                "quantile": pl.String,
                "quantile_return": pl.Float64,
                "expected_count": pl.Int64,
                "observed_count": pl.Int64,
                "coverage_ratio": pl.Float64,
                "unavailable_reason": pl.String,
            }
        )
    if {
        "_quantile_number",
        "_quantile_count",
        "_quantile_unique_valid",
    }.issubset(forward_returns.columns):
        paired = forward_returns.with_columns(
            pl.concat_str(pl.lit("q"), pl.col("_quantile_number")).alias("quantile"),
            pl.col("_quantile_count").cast(pl.Int64).alias("_count"),
            pl.when(pl.col("_quantile_unique_valid"))
            .then(pl.lit(2, dtype=pl.Int64))
            .otherwise(pl.lit(1, dtype=pl.Int64))
            .alias("_unique"),
        )
    else:
        if membership is None:
            raise RuntimeError("window quantile returns require rank lineage")
        paired = forward_returns.join(
            membership.select(
                "evaluation_date",
                ASSET_ID,
                "quantile",
                "_count",
                "_unique",
            ),
            on=["evaluation_date", ASSET_ID],
            how="left",
        )
    if paired.is_empty():
        return _empty_window_metric(
            {
                "quantile": pl.String,
                "quantile_return": pl.Float64,
                "expected_count": pl.Int64,
                "observed_count": pl.Int64,
                "coverage_ratio": pl.Float64,
                "unavailable_reason": pl.String,
            }
        )
    grouped = (
        paired.drop_nulls("quantile")
        .group_by(*_RETURN_GROUP_COLUMNS, "quantile")
        .agg(
            pl.len().alias("expected_count"),
            pl.col("forward_return").is_not_null().sum().alias("observed_count"),
            pl.col("forward_return").mean().alias("_mean_return"),
            pl.first("_count").alias("_count"),
            pl.first("_unique").alias("_unique"),
        )
        .with_columns(
            (pl.col("observed_count") / pl.col("expected_count")).alias(
                "coverage_ratio"
            )
        )
        .with_columns(
            pl.when(
                (pl.col("_count") >= quantiles)
                & (pl.col("_unique") >= 2)
                & (pl.col("observed_count") == pl.col("expected_count"))
            )
            .then(pl.col("_mean_return"))
            .otherwise(None)
            .alias("quantile_return"),
            pl.when((pl.col("_count") < quantiles) | (pl.col("_unique") < 2))
            .then(pl.lit("quantile curve requires ten non-constant ranks"))
            .when(pl.col("observed_count") != pl.col("expected_count"))
            .then(pl.lit("forward-return coverage is incomplete"))
            .otherwise(None)
            .alias("unavailable_reason"),
        )
        .select(
            *_RETURN_GROUP_COLUMNS,
            "quantile",
            "quantile_return",
            "expected_count",
            "observed_count",
            "coverage_ratio",
            "unavailable_reason",
        )
        .sort(["evaluation_date", "window_id", "quantile"])
    )
    return grouped


def quantile_curve_structure(
    quantile_returns: pl.DataFrame,
    *,
    quantiles: int = 10,
) -> pl.DataFrame:
    """Summarize ordering and linearity for each complete quantile curve."""

    schema = {
        "evaluation_date": pl.Date,
        "execution_date": pl.Date,
        "target_end_date": pl.Date,
        "window_kind": pl.String,
        "window_id": pl.String,
        "start_session": pl.Int64,
        "end_session": pl.Int64,
        "quantile_rank_ic": pl.Float64,
        "monotonicity": pl.Float64,
        "quantile_linearity_slope": pl.Float64,
        "quantile_linearity_r_squared": pl.Float64,
        "unavailable_reason": pl.String,
    }
    rows: list[dict[str, object]] = []
    for key, sample in quantile_returns.group_by(*_RETURN_GROUP_COLUMNS):
        ordered = sample.with_columns(
            pl.col("quantile").str.slice(1).cast(pl.Int64).alias("_number")
        ).sort("_number")
        values = ordered.get_column("quantile_return").to_list()
        complete = len(values) == quantiles and all(
            value is not None and math.isfinite(float(value)) for value in values
        )
        quantile_ic = None
        monotonicity = None
        linearity_slope = None
        linearity_r_squared = None
        reason = None
        if complete:
            signal_order = np.arange(quantiles, 0, -1, dtype=float)
            return_values = np.asarray(values, dtype=float)
            return_ranks = stats.rankdata(return_values, method="average")
            linearity_slope = float(
                np.cov(signal_order, return_values, ddof=0)[0, 1] / np.var(signal_order)
            )
            if np.unique(return_ranks).size >= 2:
                quantile_ic = float(np.corrcoef(signal_order, return_ranks)[0, 1])
                linearity_correlation = float(
                    np.corrcoef(signal_order, return_values)[0, 1]
                )
                linearity_r_squared = linearity_correlation**2
            else:
                reason = "quantile-rank IC requires non-constant returns"
            monotonicity = float(
                np.mean(np.asarray(values[:-1]) >= np.asarray(values[1:]))
            )
        else:
            reason = "complete q1-to-q10 curve required"
        metadata = dict(zip(_RETURN_GROUP_COLUMNS, key, strict=True))
        rows.append(
            {
                **metadata,
                "quantile_rank_ic": quantile_ic,
                "monotonicity": monotonicity,
                "quantile_linearity_slope": linearity_slope,
                "quantile_linearity_r_squared": linearity_r_squared,
                "unavailable_reason": reason,
            }
        )
    return (
        pl.DataFrame(rows, schema=schema).sort(["evaluation_date", "window_id"])
        if rows
        else pl.DataFrame(schema=schema)
    )


def window_factor_returns(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    standardization: Literal["none", "cross_sectional_zscore"] = "none",
) -> pl.DataFrame:
    """Estimate per-date cross-sectional OLS slopes for every window.

    ``cross_sectional_zscore`` uses a population standard deviation (``ddof=0``)
    on each evaluation-date cross-section, so the slope is expressed per one
    cross-sectional standard deviation.  The default preserves the historical
    raw-factor contract.
    """

    if standardization not in {"none", "cross_sectional_zscore"}:
        raise ValueError(f"unsupported factor standardization: {standardization}")

    normalized = _validate_scheduled_factor_frame(factor)
    window_groups = None
    if standardization == "cross_sectional_zscore":
        window_groups = forward_returns.select(*_RETURN_GROUP_COLUMNS).unique()
        normalized = normalized.with_columns(
            pl.len().over("evaluation_date").alias("_factor_count"),
            pl.col("factor").mean().over("evaluation_date").alias("_factor_mean"),
            pl.col("factor").std(ddof=0).over("evaluation_date").alias("_factor_std"),
        ).with_columns(
            pl.when(
                (pl.col("_factor_count") >= 3)
                & pl.col("_factor_std").is_finite()
                & (pl.col("_factor_std") > 0)
            )
            .then((pl.col("factor") - pl.col("_factor_mean")) / pl.col("_factor_std"))
            .otherwise(None)
            .alias("factor")
        )
    paired = forward_returns.join(
        normalized.select("evaluation_date", ASSET_ID, "factor"),
        on=["evaluation_date", ASSET_ID],
        how="left",
    ).drop_nulls(["factor", "forward_return"])
    if paired.is_empty():
        if window_groups is not None:
            return window_groups.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("factor_return"),
                pl.lit(0, dtype=pl.Int64).alias("sample_size"),
            ).sort(["evaluation_date", "window_id"])
        return _empty_window_metric(
            {"factor_return": pl.Float64, "sample_size": pl.Int64}
        )
    result = (
        paired.group_by(*_RETURN_GROUP_COLUMNS)
        .agg(
            pl.len().alias("sample_size"),
            pl.col("factor").n_unique().alias("_factor_values"),
            pl.col("factor").var().alias("_factor_variance"),
            pl.cov("factor", "forward_return").alias("_covariance"),
        )
        .with_columns(
            pl.when(
                (pl.col("sample_size") >= 3)
                & (pl.col("_factor_values") >= 2)
                & (pl.col("_factor_variance") > 0)
            )
            .then(pl.col("_covariance") / pl.col("_factor_variance"))
            .otherwise(None)
            .alias("factor_return")
        )
        .select(*_RETURN_GROUP_COLUMNS, "factor_return", "sample_size")
        .sort(["evaluation_date", "window_id"])
    )
    if window_groups is not None:
        result = window_groups.join(
            result,
            on=list(_RETURN_GROUP_COLUMNS),
            how="left",
        ).with_columns(pl.col("sample_size").fill_null(0))
    return result.sort(["evaluation_date", "window_id"])


def signal_rank_persistence(
    factor: pl.DataFrame,
    *,
    calendar: pl.DataFrame,
    horizons: Sequence[int] = SIGNAL_PERSISTENCE_HORIZONS,
    progress: DailyDiagnosticProgress | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Calculate same-universe rank autocorrelation and a grid half-life band."""

    normalized = _validate_scheduled_factor_frame(factor)
    sessions = _session_dates(
        calendar,
        normalized.rename({"factor": "price"}).select(
            pl.col("evaluation_date").alias(TIME), ASSET_ID, "price"
        ),
    )
    resolved_horizons = tuple(int(horizon) for horizon in horizons)
    if any(horizon < 1 for horizon in resolved_horizons):
        raise ValueError("persistence horizons must be positive")
    series_schema = {
        "evaluation_date": pl.Date,
        "target_date": pl.Date,
        "horizon_sessions": pl.Int64,
        "rank_autocorrelation": pl.Float64,
        "sample_size": pl.Int64,
    }
    session_ordinals = pl.DataFrame(
        {
            "evaluation_date": sessions,
            "_session_ordinal": range(len(sessions)),
        },
        schema={"evaluation_date": pl.Date, "_session_ordinal": pl.Int64},
    )
    indexed = normalized.join(
        session_ordinals,
        on="evaluation_date",
        how="inner",
    )
    current = indexed.select("evaluation_date", ASSET_ID, "factor", "_session_ordinal")
    future = indexed.select(
        pl.col("evaluation_date").alias("target_date"),
        ASSET_ID,
        pl.col("factor").alias("future_factor"),
        pl.col("_session_ordinal").alias("_target_ordinal"),
    )
    date_chunks: list[tuple[int, int]] = []
    chunk_start: int | None = None
    chunk_end: int | None = None
    chunk_rows = 0
    for ordinal, row_count in (
        current.group_by("_session_ordinal").len().sort("_session_ordinal").iter_rows()
    ):
        resolved_ordinal = int(ordinal)
        resolved_count = int(row_count)
        if (
            chunk_start is not None
            and chunk_rows + resolved_count > _PERSISTENCE_PAIR_ROW_BUDGET
        ):
            assert chunk_end is not None
            date_chunks.append((chunk_start, chunk_end))
            chunk_start = resolved_ordinal
            chunk_rows = 0
        if chunk_start is None:
            chunk_start = resolved_ordinal
        chunk_end = resolved_ordinal
        chunk_rows += resolved_count
    if chunk_start is not None:
        assert chunk_end is not None
        date_chunks.append((chunk_start, chunk_end))
    series_frames: list[pl.DataFrame] = []
    total = len(resolved_horizons)
    for completed, horizon in enumerate(resolved_horizons, start=1):
        for chunk_start, chunk_end in date_chunks:
            paired = (
                current.filter(
                    pl.col("_session_ordinal").is_between(
                        chunk_start,
                        chunk_end,
                    )
                )
                .with_columns(
                    (pl.col("_session_ordinal") + horizon).alias("_target_ordinal")
                )
                .join(
                    future.filter(
                        pl.col("_target_ordinal").is_between(
                            chunk_start + horizon,
                            chunk_end + horizon,
                        )
                    ),
                    on=["_target_ordinal", ASSET_ID],
                    how="inner",
                )
                .with_columns(
                    pl.lit(horizon, dtype=pl.Int64).alias("horizon_sessions"),
                    pl.col("factor")
                    .rank("average")
                    .over("evaluation_date")
                    .alias("_current_rank"),
                    pl.col("future_factor")
                    .rank("average")
                    .over("evaluation_date")
                    .alias("_future_rank"),
                )
            )
            if not paired.is_empty():
                series_frames.append(
                    paired.group_by(
                        "evaluation_date", "target_date", "horizon_sessions"
                    ).agg(
                        _safe_corr("_current_rank", "_future_rank").alias(
                            "rank_autocorrelation"
                        ),
                        pl.len().alias("sample_size"),
                    )
                )
        if progress is not None:
            progress("signal_autocorrelation", completed, total)
    series = (
        pl.concat(series_frames, how="vertical_relaxed").sort(
            ["evaluation_date", "horizon_sessions"]
        )
        if series_frames
        else pl.DataFrame(schema=series_schema)
    )
    return series, summarize_signal_persistence(series, resolved_horizons)


def hac_mean_test(
    values: object,
    *,
    window_width: int,
    null_mean: float = 0.0,
) -> HACMeanTest:
    """Run a two-sided Bartlett Newey-West mean test."""

    if window_width < 1:
        raise ValueError("window_width must be positive")
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=float,
    )
    sample_size = int(finite.size)
    mean = float(finite.mean()) if sample_size else None
    lag = max(
        window_width - 1,
        math.floor(4.0 * (sample_size / 100.0) ** (2.0 / 9.0)),
    )
    if sample_size < 2:
        return HACMeanTest(
            mean,
            None,
            None,
            None,
            None,
            None,
            sample_size,
            lag,
            "at least two samples required",
        )
    if lag >= sample_size:
        return HACMeanTest(
            mean,
            None,
            None,
            None,
            None,
            None,
            sample_size,
            lag,
            "sample size must exceed the HAC lag",
        )
    centered = finite - float(mean)
    long_run_variance = float(np.dot(centered, centered) / sample_size)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / sample_size)
        bartlett_weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * covariance
    if not math.isfinite(long_run_variance) or long_run_variance <= 0.0:
        return HACMeanTest(
            mean,
            None,
            None,
            None,
            None,
            None,
            sample_size,
            lag,
            "HAC long-run variance is not positive",
        )
    standard_error = math.sqrt(long_run_variance / sample_size)
    t_value = (float(mean) - null_mean) / standard_error
    critical = float(stats.t.ppf(0.975, df=sample_size - 1))
    p_value = float(2.0 * stats.t.sf(abs(t_value), df=sample_size - 1))
    return HACMeanTest(
        float(mean),
        standard_error,
        t_value,
        p_value,
        float(mean) - critical * standard_error,
        float(mean) + critical * standard_error,
        sample_size,
        lag,
    )


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    """Return monotone Benjamini-Hochberg q-values for finite p-values."""

    finite = [
        (index, float(value))
        for index, value in enumerate(p_values)
        if value is not None and math.isfinite(float(value))
    ]
    result: list[float | None] = [None] * len(p_values)
    if not finite:
        return result
    ordered = sorted(finite, key=lambda item: item[1])
    count = len(ordered)
    adjusted = [0.0] * count
    running = 1.0
    for position in range(count - 1, -1, -1):
        _, value = ordered[position]
        candidate = min(1.0, value * count / (position + 1))
        running = min(running, candidate)
        adjusted[position] = running
    for (index, _), value in zip(ordered, adjusted, strict=True):
        result[index] = value
    return result


def non_overlapping_cohort_statistics(
    dates: Sequence[date],
    values: Sequence[float | None],
    *,
    window_width: int,
) -> dict[str, object]:
    """Summarize every staggered non-overlapping cohort."""

    if window_width < 1:
        raise ValueError("window_width must be positive")
    if len(dates) != len(values):
        raise ValueError("dates and values must have equal length")
    ordered_dates = sorted(set(dates))
    ordinals = {value: index for index, value in enumerate(ordered_dates)}
    cohorts: list[list[float]] = [[] for _ in range(window_width)]
    all_values: list[float] = []
    for current_date, value in zip(dates, values, strict=True):
        if value is None or not math.isfinite(float(value)):
            continue
        observation = float(value)
        cohorts[ordinals[current_date] % window_width].append(observation)
        all_values.append(observation)
    means = [float(np.mean(cohort)) for cohort in cohorts if cohort]
    overall_mean = float(np.mean(all_values)) if all_values else None
    same_sign = None
    if means and overall_mean is not None and overall_mean != 0.0:
        same_sign = float(
            np.mean(
                [
                    math.copysign(1.0, value) == math.copysign(1.0, overall_mean)
                    for value in means
                    if value != 0.0
                ]
            )
        )
    return {
        "cohort_count": len(means),
        "cohort_mean_median": float(np.median(means)) if means else None,
        "cohort_mean_min": min(means) if means else None,
        "cohort_mean_max": max(means) if means else None,
        "cohort_same_sign_ratio": same_sign,
    }


def build_statistical_inference(
    *,
    ic: pl.DataFrame,
    book_returns: pl.DataFrame,
    tail_returns: pl.DataFrame,
    quantile_structure: pl.DataFrame,
    factor_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Build HAC, BH, and staggered-cohort inference for every metric family."""

    sources = (
        (ic, "pearson_ic", "pearson_ic"),
        (ic, "spearman_ic", "spearman_ic"),
        (book_returns, "book_return", "book_return"),
        (tail_returns, "tail_return", "tail_return"),
        (quantile_structure, "quantile_rank_ic", "quantile_rank_ic"),
        (factor_returns, "factor_return", "cross_section_regression"),
    )
    rows: list[dict[str, object]] = []
    for frame, column, metric in sources:
        if frame.is_empty() or column not in frame.columns:
            continue
        for key, sample in frame.group_by(*_WINDOW_COLUMNS):
            metadata = dict(zip(_WINDOW_COLUMNS, key, strict=True))
            width = int(metadata["end_session"]) - int(metadata["start_session"]) + 1
            ordered = sample.sort("evaluation_date")
            test = hac_mean_test(
                ordered.get_column(column).to_list(),
                window_width=width,
            )
            cohorts = non_overlapping_cohort_statistics(
                ordered.get_column("evaluation_date").to_list(),
                ordered.get_column(column).to_list(),
                window_width=width,
            )
            rows.append(
                {
                    "metric": metric,
                    **metadata,
                    "window_width": width,
                    "mean": test.mean,
                    "hac_standard_error": test.standard_error,
                    "hac_t": test.t_value,
                    "p_value": test.p_value,
                    "q_value": None,
                    "confidence_low": test.confidence_low,
                    "confidence_high": test.confidence_high,
                    "sample_size": test.sample_size,
                    "hac_lag": test.lag,
                    "unavailable_reason": test.reason,
                    **cohorts,
                }
            )
    if not rows:
        return pl.DataFrame(schema=_inference_schema())
    result = pl.DataFrame(rows, schema=_inference_schema())
    adjusted: list[pl.DataFrame] = []
    for _metric, family in result.group_by("metric"):
        values = family.get_column("p_value").to_list()
        adjusted.append(
            family.with_columns(
                pl.Series(
                    "q_value",
                    benjamini_hochberg(values),
                    dtype=pl.Float64,
                )
            )
        )
    return pl.concat(adjusted).sort(["metric", "window_kind", "end_session"])


def _prepare_daily_diagnostics(
    signals: ScheduledPrediction,
    prices: pl.DataFrame,
    *,
    calendar: pl.DataFrame | None,
    quantiles: int,
    prediction_frame: pl.DataFrame | None = None,
) -> _PreparedDailyDiagnostics:
    """Validate and derive the large immutable inputs shared by daily research."""

    market = validate_prices(prices)
    factor = _scheduled_factor(signals, prediction_frame=prediction_frame).join(
        market.select(
            pl.col(TIME).alias("execution_date"),
            ASSET_ID,
            pl.col("price").alias("_execution_price"),
        ),
        on=["execution_date", ASSET_ID],
        how="left",
    )
    resolved_calendar = (
        calendar if calendar is not None else market.select(TIME).unique()
    )
    return _PreparedDailyDiagnostics(
        factor=factor,
        market=market,
        calendar=resolved_calendar,
        sessions=tuple(_session_dates(resolved_calendar, market)),
        market_sessions=tuple(_session_dates(None, market)),
    )


def _prepare_daily_weights(
    prepared: _PreparedDailyDiagnostics,
    *,
    quantiles: int,
) -> _PreparedDailyDiagnostics:
    """Derive large rank-weight tables only when window aggregation begins."""

    book_active = _centered_rank_book_active_lineage(prepared.factor)
    quantile_membership = _quantile_membership(
        prepared.factor,
        quantiles=quantiles,
    )
    tail = _gross_one_tail_weights_with_membership(
        quantile_membership,
        quantiles=quantiles,
    ).filter(pl.col("tail_weight").is_null() | (pl.col("tail_weight") != 0.0))
    factor_columns = [
        "evaluation_date",
        "execution_date",
        ASSET_ID,
        "factor",
    ]
    if "_execution_price" in quantile_membership.columns:
        factor_columns.append("_execution_price")
    enriched_factor = quantile_membership.select(
        *factor_columns,
        pl.col("quantile").str.slice(1).cast(pl.UInt8).alias("_quantile_number"),
        pl.col("_count").cast(pl.UInt32).alias("_quantile_count"),
        (pl.col("_unique") >= 2).alias("_quantile_unique_valid"),
    ).sort(["evaluation_date", ASSET_ID])
    book = _centered_rank_book_weights_prepared(quantile_membership).hstack(
        [
            (
                quantile_membership.select(
                    "evaluation_date",
                    ASSET_ID,
                    pl.when(pl.col("_unique") >= 2)
                    .then(pl.col("quantile").str.slice(1).cast(pl.UInt8))
                    .otherwise(None)
                    .alias("_quantile_number"),
                )
                .sort(["evaluation_date", ASSET_ID])
                .get_column("_quantile_number")
            ),
            book_active,
        ]
    )
    return replace(
        prepared,
        factor=enriched_factor,
        book=book,
        tail=tail,
        quantile_membership=None,
    )


def _prepare_daily_price_lookup(
    prepared: _PreparedDailyDiagnostics,
) -> _PreparedDailyDiagnostics:
    """Sort the shared as-of price lookup once for all horizon windows."""

    return replace(
        prepared,
        price_lookup=prepared.market.select(
            ASSET_ID,
            pl.col(TIME).alias("_observed_date"),
            "price",
        ).sort([ASSET_ID, "_observed_date"]),
    )


def run_prediction_horizon_diagnostics(
    signals: ScheduledPrediction,
    prices: pl.DataFrame,
    *,
    windows: Sequence[SessionWindow] = DAILY_SESSION_WINDOWS,
    calendar: pl.DataFrame | None = None,
    quantiles: int = 10,
    annualization_sessions: int = 240,
    prediction_frame: pl.DataFrame | None = None,
    factor_standardization: Literal["none", "cross_sectional_zscore"] = "none",
) -> PredictionHorizonDiagnostics:
    """Run fixed-window diagnostics while retaining one label window at a time."""

    if annualization_sessions <= 0:
        raise ValueError("annualization_sessions must be positive")
    resolved_windows = _validate_windows(windows)
    prepared = _prepare_daily_diagnostics(
        signals,
        prices,
        calendar=calendar,
        quantiles=quantiles,
        prediction_frame=prediction_frame,
    )
    prepared = _prepare_daily_weights(prepared, quantiles=quantiles)
    prepared = _prepare_daily_price_lookup(prepared)
    return _run_prediction_horizon_diagnostics_prepared(
        prepared,
        windows=resolved_windows,
        quantiles=quantiles,
        annualization_sessions=annualization_sessions,
        factor_standardization=factor_standardization,
    )


def _run_prediction_horizon_diagnostics_prepared(
    prepared: _PreparedDailyDiagnostics,
    *,
    windows: Sequence[SessionWindow],
    quantiles: int,
    annualization_sessions: int,
    factor_standardization: Literal["none", "cross_sectional_zscore"] = "none",
    signal_persistence: pl.DataFrame | None = None,
) -> PredictionHorizonDiagnostics:
    """Aggregate window diagnostics from a caller-owned prepared context."""

    if prepared.book is None or prepared.tail is None:
        raise RuntimeError("daily horizon diagnostics require prepared rank weights")

    coverage_frames: list[pl.DataFrame] = []
    ic_frames: list[pl.DataFrame] = []
    book_return_frames: list[pl.DataFrame] = []
    tail_return_frames: list[pl.DataFrame] = []
    quantile_return_frames: list[pl.DataFrame] = []
    factor_return_frames: list[pl.DataFrame] = []
    max_window_forward_rows = 0
    for window in windows:
        forward, window_coverage = _session_window_forward_returns_from_frames(
            prepared.factor,
            prepared.market,
            windows=(window,),
            sessions=prepared.sessions,
            price_lookup=prepared.price_lookup,
            compact_window_keys=True,
        )
        max_window_forward_rows = max(max_window_forward_rows, forward.height)
        coverage_frames.append(window_coverage)
        ic_frames.append(
            _restore_window_key_dtypes(
                window_information_coefficients(prepared.factor, forward)
            )
        )
        book_return_frames.append(
            _restore_window_key_dtypes(window_book_returns(prepared.book, forward))
        )
        tail_return_frames.append(
            _restore_window_key_dtypes(window_tail_returns(prepared.tail, forward))
        )
        quantile_return_frames.append(
            _restore_window_key_dtypes(
                _window_quantile_forward_returns_with_membership(
                    prepared.quantile_membership,
                    forward,
                    quantiles=quantiles,
                )
            )
        )
        factor_return_frames.append(
            _restore_window_key_dtypes(
                window_factor_returns(
                    prepared.factor,
                    forward,
                    standardization=factor_standardization,
                )
            )
        )
    coverage = pl.concat(coverage_frames, how="diagonal_relaxed").sort(
        ["evaluation_date", "window_id"]
    )
    ic = pl.concat(ic_frames, how="diagonal_relaxed").sort(
        ["evaluation_date", "window_id"]
    )
    book_returns = pl.concat(book_return_frames, how="diagonal_relaxed").sort(
        ["evaluation_date", "window_id"]
    )
    tail_returns = pl.concat(tail_return_frames, how="diagonal_relaxed").sort(
        ["evaluation_date", "window_id"]
    )
    quantile_returns = pl.concat(quantile_return_frames, how="diagonal_relaxed").sort(
        ["evaluation_date", "window_id", "quantile"]
    )
    factor_returns = pl.concat(factor_return_frames, how="diagonal_relaxed").sort(
        ["evaluation_date", "window_id"]
    )
    structure = quantile_curve_structure(quantile_returns, quantiles=quantiles)
    if signal_persistence is None:
        persistence, persistence_summary = signal_rank_persistence(
            prepared.factor, calendar=prepared.calendar
        )
    else:
        persistence = signal_persistence.filter(
            pl.col("horizon_sessions").is_in(SIGNAL_PERSISTENCE_HORIZONS)
        ).sort(["evaluation_date", "horizon_sessions"])
        persistence_summary = summarize_signal_persistence(
            persistence,
            SIGNAL_PERSISTENCE_HORIZONS,
        )
    return PredictionHorizonDiagnostics(
        coverage=coverage,
        ic=ic,
        ic_summary=summarize_window_ic(
            ic, annualization_sessions=annualization_sessions
        ),
        book_returns=book_returns,
        tail_returns=tail_returns,
        quantile_forward_returns=quantile_returns,
        quantile_structure=structure,
        factor_returns=factor_returns,
        signal_persistence=persistence,
        signal_persistence_summary=persistence_summary,
        statistical_inference=build_statistical_inference(
            ic=ic,
            book_returns=book_returns,
            tail_returns=tail_returns,
            quantile_structure=structure,
            factor_returns=factor_returns,
        ),
        max_window_forward_rows=max_window_forward_rows,
    )


def rolling_window_information_coefficients(
    ic: pl.DataFrame,
    *,
    observations: int = DAILY_ROLLING_IC_OBSERVATIONS,
) -> pl.DataFrame:
    """Calculate causal rolling IC means over a fixed count of valid values.

    Null IC observations do not consume the window.  A value is published only
    on a date with a finite IC and only after ``observations`` valid values are
    available for that exact window and method.
    """

    if observations < 1:
        raise ValueError("rolling IC observations must be positive")
    required = {
        "evaluation_date",
        *_WINDOW_COLUMNS,
        "pearson_ic",
        "spearman_ic",
    }
    missing = sorted(required - set(ic.columns))
    if missing:
        raise ValueError(f"IC series is missing required columns: {missing}")
    if ic.is_empty():
        return pl.DataFrame(schema=_rolling_ic_schema())
    group_columns = [*_WINDOW_COLUMNS, "method"]
    keys = ["evaluation_date", *group_columns]
    long = (
        ic.select(
            "evaluation_date",
            *_WINDOW_COLUMNS,
            "pearson_ic",
            "spearman_ic",
        )
        .unpivot(
            on=["pearson_ic", "spearman_ic"],
            index=["evaluation_date", *_WINDOW_COLUMNS],
            variable_name="method",
            value_name="_value",
        )
        .with_columns(
            pl.col("method").replace_strict(
                {"pearson_ic": "pearson", "spearman_ic": "spearman"}
            ),
            pl.col("_value").is_finite().fill_null(False).alias("_is_valid"),
        )
        .sort([*group_columns, "evaluation_date"])
        .with_columns(
            pl.col("_is_valid")
            .cast(pl.Int64)
            .cum_sum()
            .over(*group_columns)
            .clip(upper_bound=observations)
            .alias("rolling_observations")
        )
    )
    rolling = (
        long.filter("_is_valid")
        .with_columns(
            pl.col("_value")
            .rolling_mean(
                window_size=observations,
                min_samples=observations,
            )
            .over(*group_columns)
            .alias("rolling_ic")
        )
        .select(*keys, "rolling_ic")
    )
    return (
        long.join(rolling, on=keys, how="left")
        .select(
            "evaluation_date",
            *_WINDOW_COLUMNS,
            "method",
            "rolling_ic",
            "rolling_observations",
        )
        .sort(["window_kind", "end_session", "method", "evaluation_date"])
    )


def implied_signal_half_life(
    lag: int,
    rank_autocorrelation: float | None,
) -> float | None:
    """Infer half-life from one lag/rank-autocorrelation observation."""

    if lag < 1:
        raise ValueError("half-life lag must be positive")
    if rank_autocorrelation is None:
        return None
    correlation = float(rank_autocorrelation)
    if not math.isfinite(correlation) or not 0.0 < correlation < 1.0:
        return None
    return -float(lag) * math.log(2.0) / math.log(correlation)


def run_daily_rank_path_diagnostics(
    signals: ScheduledPrediction,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
    calendar: pl.DataFrame | None = None,
    quantiles: int = 10,
    horizon_ic: pl.DataFrame | None = None,
    lead_lags: Sequence[int] = DAILY_BOOK_LEAD_LAGS,
    alpha_return_lags: Sequence[int] = DAILY_ALPHA_RETURN_LAGS,
    autocorrelation_lags: Sequence[int] = DAILY_SUMMARY_AUTOCORRELATION_LAGS,
    rolling_observations: int = DAILY_ROLLING_IC_OBSERVATIONS,
    progress: DailyDiagnosticProgress | None = None,
    prediction_frame: pl.DataFrame | None = None,
) -> DailyRankPathDiagnostics:
    """Run daily paths from an independently prepared daily input context."""

    prepared = _prepare_daily_diagnostics(
        signals,
        prices,
        calendar=calendar,
        quantiles=quantiles,
        prediction_frame=prediction_frame,
    )
    prepared = _prepare_daily_weights(prepared, quantiles=quantiles)
    daily_forward_returns = _prepare_price_data(
        prepared.market,
        inputs_sorted=True,
    ).forward_returns
    prepared = replace(prepared, quantile_membership=None)
    return _run_daily_rank_path_diagnostics_prepared(
        prepared,
        daily_forward_returns=daily_forward_returns,
        quantiles=quantiles,
        config=config,
        execution_availability=execution_availability,
        slippage_rates=slippage_rates,
        horizon_ic=horizon_ic,
        lead_lags=lead_lags,
        alpha_return_lags=alpha_return_lags,
        autocorrelation_lags=autocorrelation_lags,
        rolling_observations=rolling_observations,
        progress=progress,
    )


def run_daily_prediction_diagnostics(
    signals: ScheduledPrediction,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
    calendar: pl.DataFrame | None = None,
    windows: Sequence[SessionWindow] = DAILY_SESSION_WINDOWS,
    quantiles: int = 10,
    annualization_sessions: int = 240,
    lead_lags: Sequence[int] = DAILY_BOOK_LEAD_LAGS,
    alpha_return_lags: Sequence[int] = DAILY_ALPHA_RETURN_LAGS,
    autocorrelation_lags: Sequence[int] = DAILY_SUMMARY_AUTOCORRELATION_LAGS,
    rolling_observations: int = DAILY_ROLLING_IC_OBSERVATIONS,
    progress: DailyDiagnosticProgress | None = None,
    prediction_frame: pl.DataFrame | None = None,
    factor_standardization: Literal["none", "cross_sectional_zscore"] = "none",
) -> DailyPredictionDiagnostics:
    """Run all daily prediction diagnostics from one validated input context."""

    if annualization_sessions <= 0:
        raise ValueError("annualization_sessions must be positive")
    resolved_windows = _validate_windows(windows)
    prepared = _prepare_daily_diagnostics(
        signals,
        prices,
        calendar=calendar,
        quantiles=quantiles,
        prediction_frame=prediction_frame,
    )
    resolved_autocorrelation_lags = _validate_positive_lags(
        autocorrelation_lags,
        label="autocorrelation",
    )
    persistence_lags = tuple(
        sorted(set(resolved_autocorrelation_lags) | set(SIGNAL_PERSISTENCE_HORIZONS))
    )
    signal_autocorrelation, _ = signal_rank_persistence(
        prepared.factor,
        calendar=prepared.calendar,
        horizons=persistence_lags,
        progress=progress,
    )
    prepared = _prepare_daily_weights(prepared, quantiles=quantiles)
    prepared = _prepare_daily_price_lookup(prepared)
    horizons = _run_prediction_horizon_diagnostics_prepared(
        prepared,
        windows=resolved_windows,
        quantiles=quantiles,
        annualization_sessions=annualization_sessions,
        factor_standardization=factor_standardization,
        signal_persistence=signal_autocorrelation,
    )
    if progress is not None:
        progress("prediction_horizons", 1, 1)
    path_autocorrelation = signal_autocorrelation.filter(
        pl.col("horizon_sessions").is_in(resolved_autocorrelation_lags)
    ).sort(["evaluation_date", "horizon_sessions"])
    path_prepared = replace(
        prepared,
        factor=pl.DataFrame(schema=prepared.factor.schema),
        calendar=pl.DataFrame(schema=prepared.calendar.schema),
        sessions=(),
        book=prepared.book.select(
            "execution_date",
            ASSET_ID,
            "book_weight",
            "_quantile_number",
        ),
        tail=prepared.tail.select(
            "execution_date",
            ASSET_ID,
            "tail_weight",
        ),
        price_lookup=None,
    )
    del prepared, signal_autocorrelation
    daily_forward_returns = _prepare_price_data(
        path_prepared.market,
        inputs_sorted=True,
    ).forward_returns
    paths = _run_daily_rank_path_diagnostics_prepared(
        path_prepared,
        daily_forward_returns=daily_forward_returns,
        quantiles=quantiles,
        config=config,
        execution_availability=execution_availability,
        slippage_rates=slippage_rates,
        horizon_ic=horizons.ic,
        lead_lags=lead_lags,
        alpha_return_lags=alpha_return_lags,
        autocorrelation_lags=autocorrelation_lags,
        rolling_observations=rolling_observations,
        progress=progress,
        signal_autocorrelation=path_autocorrelation,
    )
    return DailyPredictionDiagnostics(horizons=horizons, paths=paths)


def _run_daily_rank_path_diagnostics_prepared(
    prepared_context: _PreparedDailyDiagnostics,
    *,
    daily_forward_returns: pl.DataFrame,
    quantiles: int,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None,
    slippage_rates: pl.DataFrame | None,
    horizon_ic: pl.DataFrame | None,
    lead_lags: Sequence[int],
    alpha_return_lags: Sequence[int],
    autocorrelation_lags: Sequence[int],
    rolling_observations: int,
    progress: DailyDiagnosticProgress | None,
    signal_autocorrelation: pl.DataFrame | None = None,
) -> DailyRankPathDiagnostics:
    """Run daily Book/Tail paths and the inputs needed by daily summaries.

    Lead-lag values are signed trading offsets: negative values trade before
    the signal's mapped execution session and are intentionally non-PIT;
    positive values delay execution.  Every lead-lag path is restricted to
    exactly the same return sessions.  These paths are proportional-cost
    factor returns and never construct a capital account.
    """

    resolved_lags = _validate_lead_lags(lead_lags)
    resolved_alpha_return_lags = _validate_lead_lags(alpha_return_lags)
    resolved_autocorrelation_lags = _validate_positive_lags(
        autocorrelation_lags,
        label="autocorrelation",
    )
    market = prepared_context.market
    factor = prepared_context.factor
    if prepared_context.book is None or prepared_context.tail is None:
        raise RuntimeError("daily path diagnostics require prepared rank weights")
    book_weights = _path_weight_frame(prepared_context.book, column="book_weight")
    tail_weights = _path_weight_frame(prepared_context.tail, column="tail_weight")
    market_sessions = prepared_context.market_sessions
    common_lead_lag_dates = _common_lead_lag_dates(
        book_weights,
        sessions=market_sessions,
        lead_lags=resolved_lags,
    )
    resolved_availability = (
        None
        if execution_availability is None
        else validate_execution_availability(execution_availability)
    )
    resolved_slippage = (
        None if slippage_rates is None else validate_slippage_rates(slippage_rates)
    )
    base_weights = {
        label: frame
        for label, frame in (("book", book_weights), ("tail", tail_weights))
        if not frame.is_empty()
    }
    book_daily_returns, quantile_returns, book_deltas = _research_book_quantile_returns(
        prepared_context.book,
        daily_forward_returns,
        weights=book_weights,
        config=config,
        slippage_rates=resolved_slippage,
        market_sessions=market_sessions,
        quantiles=quantiles,
    )
    tail_deltas = _requested_snapshot_weight_deltas(tail_weights)
    tail_daily_returns = _research_path_returns(
        tail_weights,
        daily_forward_returns,
        config=config,
        slippage_rates=resolved_slippage,
        weight_deltas=tail_deltas,
    )
    executed_turnover = _sparse_executed_turnover(
        book_weights,
        market,
        daily_forward_returns,
        execution_availability=resolved_availability,
        retry_blocked=config.retry_blocked_orders,
    )
    if progress is not None:
        progress("book_tail_paths", 1, 1)
    book_turnover = _daily_turnover_frame(
        book_weights,
        executed_turnover,
        requested_deltas=book_deltas,
    )
    lead_lag_returns = _stream_lead_lag_returns(
        book_weights,
        daily_forward_returns,
        config=config,
        slippage_rates=resolved_slippage,
        sessions=market_sessions,
        lead_lags=resolved_lags,
        common_dates=common_lead_lag_dates,
        progress=progress,
        base_weight_deltas=book_deltas,
    )
    alpha_return_common_dates = _common_named_lead_lag_dates(
        base_weights,
        sessions=market_sessions,
        lead_lags=resolved_alpha_return_lags,
    )
    alpha_return_lag_returns = _stream_named_lead_lag_returns(
        base_weights,
        daily_forward_returns,
        config=config,
        slippage_rates=resolved_slippage,
        sessions=market_sessions,
        lead_lags=resolved_alpha_return_lags,
        common_dates=alpha_return_common_dates,
        progress=progress,
        base_weight_deltas={"book": book_deltas, "tail": tail_deltas},
    )

    if signal_autocorrelation is None:
        autocorrelation, _ = signal_rank_persistence(
            factor,
            calendar=prepared_context.calendar,
            horizons=resolved_autocorrelation_lags,
            progress=progress,
        )
    else:
        autocorrelation = signal_autocorrelation
    resolved_ic = horizon_ic
    if resolved_ic is None:
        ic_frames = []
        for window in DAILY_SESSION_WINDOWS:
            forward, _ = _session_window_forward_returns_from_frames(
                factor,
                market,
                windows=(window,),
                sessions=prepared_context.sessions,
            )
            ic_frames.append(window_information_coefficients(factor, forward))
        resolved_ic = pl.concat(
            ic_frames,
            how="diagonal_relaxed",
        ).sort(["evaluation_date", "window_id"])
    rolling_ic = rolling_window_information_coefficients(
        resolved_ic,
        observations=rolling_observations,
    )

    return DailyRankPathDiagnostics(
        book_daily_returns=book_daily_returns,
        tail_daily_returns=tail_daily_returns,
        quantile_returns=quantile_returns,
        book_turnover=book_turnover,
        book_lead_lag_returns=lead_lag_returns,
        alpha_return_lag_returns=alpha_return_lag_returns,
        signal_autocorrelation=autocorrelation,
        rolling_ic=rolling_ic,
    )


def _validate_lead_lags(lead_lags: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in lead_lags)
    if not values:
        raise ValueError("at least one lead-lag offset is required")
    if len(set(values)) != len(values):
        raise ValueError("lead-lag offsets must be unique")
    if 0 not in values:
        raise ValueError("lead-lag offsets must include zero")
    return tuple(sorted(values))


def _validate_positive_lags(
    lags: Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    values = tuple(int(value) for value in lags)
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{label} lags must be positive")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} lags must be unique")
    return tuple(sorted(values))


def _research_book_quantile_returns(
    book: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    weights: pl.DataFrame,
    config: BacktestConfig,
    slippage_rates: pl.DataFrame | None,
    market_sessions: Sequence[date],
    quantiles: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Aggregate Book and all daily quantiles from one asset-return join.

    Quantile returns use the same frozen/recovery forward-return labels as the
    Book.  Grouping by date and quantile first also yields the Book contribution
    by bucket, so no second full asset join is required.
    """

    required = {
        "execution_date",
        ASSET_ID,
        "book_weight",
        "_quantile_number",
    }
    missing = sorted(required - set(book.columns))
    if missing:
        raise ValueError(
            f"Book quantile lineage is missing required columns: {missing}"
        )
    if book.is_empty():
        return (
            pl.DataFrame(schema=_daily_return_schema()),
            pl.DataFrame(schema=_daily_quantile_return_schema()),
            _requested_snapshot_weight_deltas(weights),
        )

    timeline = (
        book.drop_nulls("book_weight")
        .select(pl.col("execution_date").alias(TIME))
        .unique()
        .sort(TIME)
    )
    grouped = _aggregate_daily_book_quantile_chunks(
        book,
        forward_returns,
        quantiles=quantiles,
    )
    gross = grouped.select(
        TIME,
        pl.col("_book_contribution").alias("gross_return"),
    )
    aggregated = pl.concat(
        [
            grouped.select(
                TIME,
                pl.lit(f"q{number}").alias("quantile"),
                pl.col(f"_count_q{number}").alias("constituent_count"),
                pl.col(f"_gross_q{number}").alias("_gross_return"),
            )
            for number in range(1, quantiles + 1)
        ],
        how="vertical_relaxed",
    )
    quantile_returns = _finalize_daily_quantile_returns(
        book,
        aggregated,
        market_sessions=market_sessions,
        quantiles=quantiles,
    )
    weight_deltas = _requested_snapshot_weight_deltas(weights)
    costs = _proportional_research_costs(
        weight_deltas,
        config=config,
        slippage_rates=slippage_rates,
    )
    book_returns = (
        timeline.join(gross, on=TIME, how="left")
        .join(costs, on=TIME, how="left")
        .with_columns(
            pl.col("gross_return").fill_null(0.0),
            pl.col("cost_return").fill_null(0.0),
        )
        .with_columns(
            (pl.col("gross_return") - pl.col("cost_return")).alias("net_return")
        )
        .select(TIME, "gross_return", "net_return")
        .sort(TIME)
    )
    return book_returns, quantile_returns, weight_deltas


def _aggregate_daily_book_quantile_chunks(
    book: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    """Aggregate complete date-bounded chunks without a full asset join."""

    book_runs = (
        book.group_by("execution_date", maintain_order=True)
        .len()
        .sort("execution_date")
    )
    forward_runs = forward_returns.group_by(TIME, maintain_order=True).len().sort(TIME)
    forward_offsets: dict[date, tuple[int, int]] = {}
    forward_offset = 0
    for session, row_count in forward_runs.iter_rows():
        resolved_count = int(row_count)
        forward_offsets[session] = (forward_offset, resolved_count)
        forward_offset += resolved_count

    chunks: list[tuple[int, int, tuple[date, ...]]] = []
    book_offset = 0
    chunk_offset = 0
    chunk_rows = 0
    chunk_dates: list[date] = []
    for session, row_count in book_runs.iter_rows():
        resolved_count = int(row_count)
        if (
            chunk_rows
            and chunk_rows + resolved_count > _DAILY_BOOK_QUANTILE_JOIN_ROW_BUDGET
        ):
            chunks.append((chunk_offset, chunk_rows, tuple(chunk_dates)))
            chunk_offset = book_offset
            chunk_rows = 0
            chunk_dates = []
        chunk_rows += resolved_count
        book_offset += resolved_count
        chunk_dates.append(session)
    if chunk_rows:
        chunks.append((chunk_offset, chunk_rows, tuple(chunk_dates)))

    frames: list[pl.DataFrame] = []
    for chunk_offset, chunk_rows, chunk_dates in chunks:
        forward_bounds = [
            forward_offsets[session]
            for session in chunk_dates
            if session in forward_offsets
        ]
        if forward_bounds:
            forward_start = forward_bounds[0][0]
            last_offset, last_count = forward_bounds[-1]
            forward_chunk = forward_returns.slice(
                forward_start,
                last_offset + last_count - forward_start,
            )
        else:
            forward_chunk = forward_returns.head(0)
        frames.append(
            _aggregate_daily_book_quantile_chunk(
                book.slice(chunk_offset, chunk_rows),
                forward_chunk,
                quantiles=quantiles,
            )
        )
    return pl.concat(frames).sort(TIME)


def _aggregate_daily_book_quantile_chunk(
    book: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    return (
        book.lazy()
        .select(
            pl.col("execution_date").alias(TIME),
            ASSET_ID,
            "book_weight",
            "_quantile_number",
        )
        .join(
            forward_returns.lazy().select(TIME, ASSET_ID, "forward_return"),
            on=[TIME, ASSET_ID],
            how="left",
        )
        .with_columns(pl.col("forward_return").fill_null(0.0).alias("_return"))
        .group_by(TIME, maintain_order=True)
        .agg(
            (pl.col("book_weight").fill_null(0.0) * pl.col("_return"))
            .sum()
            .alias("_book_contribution"),
            *(
                expression
                for number in range(1, quantiles + 1)
                for expression in (
                    pl.col("_return")
                    .filter(pl.col("_quantile_number") == number)
                    .mean()
                    .alias(f"_gross_q{number}"),
                    pl.col("_return")
                    .filter(pl.col("_quantile_number") == number)
                    .len()
                    .cast(pl.Int64)
                    .alias(f"_count_q{number}"),
                )
            ),
        )
        .collect()
    )


def _finalize_daily_quantile_returns(
    membership: pl.DataFrame,
    aggregated: pl.DataFrame,
    *,
    market_sessions: Sequence[date],
    quantiles: int,
) -> pl.DataFrame:
    """Add the common structural grid to aggregated daily quantile returns."""

    available_sessions = pl.DataFrame(
        {TIME: list(market_sessions[:-1])},
        schema={TIME: pl.Date},
    ).with_columns(pl.lit(True).alias("_has_next_session"))
    dates = (
        membership.group_by("execution_date", maintain_order=True)
        .agg(
            pl.len().cast(pl.Int64).alias("_count"),
            pl.col("_quantile_number").is_not_null().any().alias("_quantile_valid"),
        )
        .rename({"execution_date": TIME})
        .join(available_sessions, on=TIME, how="left")
        .with_columns(pl.col("_has_next_session").fill_null(False))
    )
    labels = pl.DataFrame(
        {"quantile": [f"q{number}" for number in range(1, quantiles + 1)]},
        schema={"quantile": pl.String},
    )
    grid = dates.join(labels, how="cross")

    unavailable_reason = (
        pl.when(pl.col("_count") < quantiles)
        .then(pl.lit(f"quantile portfolios require at least {quantiles} assets"))
        .when(~pl.col("_quantile_valid"))
        .then(pl.lit("quantile portfolios require a non-constant signal"))
        .when(~pl.col("_has_next_session"))
        .then(pl.lit("quantile portfolio return requires a following session"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    return (
        grid.join(aggregated, on=[TIME, "quantile"], how="left")
        .with_columns(unavailable_reason.alias("unavailable_reason"))
        .with_columns(
            pl.col("constituent_count").fill_null(0).cast(pl.Int64),
            pl.when(pl.col("unavailable_reason").is_null())
            .then(pl.col("_gross_return").fill_null(0.0))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("gross_return"),
            pl.col("quantile").str.slice(1).cast(pl.Int64).alias("_quantile_number"),
        )
        .select(
            TIME,
            "quantile",
            "gross_return",
            "constituent_count",
            "unavailable_reason",
            "_quantile_number",
        )
        .sort([TIME, "_quantile_number"])
        .drop("_quantile_number")
    )


def _path_weight_frame(weights: pl.DataFrame, *, column: str) -> pl.DataFrame:
    if weights.is_empty():
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64}
        )
    return (
        weights.drop_nulls(column)
        .select(
            pl.col("execution_date").alias(TIME),
            ASSET_ID,
            pl.col(column).alias("weight"),
        )
        .sort([TIME, ASSET_ID])
    )


def _common_lead_lag_dates(
    book_weights: pl.DataFrame,
    *,
    sessions: Sequence[date],
    lead_lags: Sequence[int],
) -> tuple[date, ...]:
    if book_weights.is_empty() or len(sessions) < 2:
        return ()
    positions = {session: index for index, session in enumerate(sessions)}
    source_dates = book_weights.get_column(TIME).unique().sort().to_list()
    date_sets: list[set[date]] = []
    for lag in lead_lags:
        shifted = {
            sessions[position + lag]
            for source in source_dates
            if (position := positions.get(source)) is not None
            and 0 <= position + lag < len(sessions) - 1
        }
        if not shifted:
            return ()
        date_sets.append(shifted)
    return tuple(sorted(set.intersection(*date_sets))) if date_sets else ()


def _common_named_lead_lag_dates(
    weights_by_path: Mapping[str, pl.DataFrame],
    *,
    sessions: Sequence[date],
    lead_lags: Sequence[int],
) -> tuple[date, ...]:
    """Return one fair overlap shared by every requested path and lag."""

    if not weights_by_path:
        return ()
    source_sets = [
        set(frame.get_column(TIME).unique().to_list())
        for frame in weights_by_path.values()
        if not frame.is_empty()
    ]
    if len(source_sets) != len(weights_by_path) or not source_sets:
        return ()
    shared_sources = set.intersection(*source_sets)
    if not shared_sources:
        return ()
    source_frame = pl.DataFrame(
        {
            TIME: sorted(shared_sources),
            ASSET_ID: ["__common__"] * len(shared_sources),
            "weight": [0.0] * len(shared_sources),
        },
        schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64},
    )
    return _common_lead_lag_dates(
        source_frame,
        sessions=sessions,
        lead_lags=lead_lags,
    )


def _shifted_lead_lag_weights(
    book_weights: pl.DataFrame,
    *,
    sessions: Sequence[date],
    lag: int,
    common_dates: Sequence[date],
) -> pl.DataFrame:
    if not common_dates:
        return book_weights.head(0)
    positions = {session: index for index, session in enumerate(sessions)}
    common = set(common_dates)
    mappings = [
        {"_source_time": source, TIME: target}
        for source in book_weights.get_column(TIME).unique().sort().to_list()
        if (position := positions.get(source)) is not None
        and 0 <= position + lag < len(sessions) - 1
        and (target := sessions[position + lag]) in common
    ]
    mapping = pl.DataFrame(
        mappings,
        schema={"_source_time": pl.Date, TIME: pl.Date},
    )
    if mapping.is_empty():
        return book_weights.head(0)
    return (
        book_weights.rename({TIME: "_source_time"})
        .join(mapping, on="_source_time", how="inner")
        .select(TIME, ASSET_ID, "weight")
    )


def _shifted_lead_lag_deltas(
    weights: pl.DataFrame,
    base_deltas: pl.DataFrame,
    *,
    sessions: Sequence[date],
    lag: int,
    common_dates: Sequence[date],
) -> pl.DataFrame | None:
    """Shift reusable deltas when the common sample is one contiguous snapshot run."""

    if not common_dates:
        return base_deltas.head(0)
    positions = {session: index for index, session in enumerate(sessions)}
    source_dates = weights.get_column(TIME).unique().sort().to_list()
    source_indices = {source: index for index, source in enumerate(source_dates)}
    mappings = [
        {"_source_time": sessions[positions[target] - lag], TIME: target}
        for target in common_dates
        if target in positions and 0 <= positions[target] - lag < len(sessions)
    ]
    if len(mappings) != len(common_dates):
        return None
    selected_sources = [mapping["_source_time"] for mapping in mappings]
    selected_indices = [source_indices.get(source) for source in selected_sources]
    if any(index is None for index in selected_indices):
        return None
    first_index = selected_indices[0]
    assert first_index is not None
    if selected_indices != list(
        range(first_index, first_index + len(selected_indices))
    ):
        return None
    mapping = pl.DataFrame(
        mappings,
        schema={"_source_time": pl.Date, TIME: pl.Date},
    )
    first_source = selected_sources[0]
    first_target = common_dates[0]
    shifted = (
        base_deltas.rename({TIME: "_source_time"})
        .join(mapping, on="_source_time", how="inner")
        .filter(pl.col(TIME) != first_target)
        .select(TIME, ASSET_ID, "signed_weight_delta")
    )
    initial = weights.filter(
        (pl.col(TIME) == first_source) & (pl.col("weight") != 0.0)
    ).select(
        pl.lit(first_target, dtype=pl.Date).alias(TIME),
        ASSET_ID,
        pl.col("weight").alias("signed_weight_delta"),
    )
    return pl.concat([initial, shifted]).sort([TIME, ASSET_ID])


def _research_path_returns(
    weights: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    slippage_rates: pl.DataFrame | None,
    weight_deltas: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Calculate a capital-free factor path with proportional trading costs."""

    if weights.is_empty():
        return pl.DataFrame(schema=_daily_return_schema())
    timeline = weights.select(TIME).unique().sort(TIME)
    gross = (
        weights.join(forward_returns, on=[TIME, ASSET_ID], how="left")
        .with_columns(
            (pl.col("weight") * pl.col("forward_return").fill_null(0.0)).alias(
                "_weighted_return"
            )
        )
        .group_by(TIME)
        .agg(pl.col("_weighted_return").sum().alias("gross_return"))
    )
    deltas = (
        _requested_snapshot_weight_deltas(weights)
        if weight_deltas is None
        else weight_deltas
    )
    costs = _proportional_research_costs(
        deltas,
        config=config,
        slippage_rates=slippage_rates,
    )
    return (
        timeline.join(gross, on=TIME, how="left")
        .join(costs, on=TIME, how="left")
        .with_columns(
            pl.col("gross_return").fill_null(0.0),
            pl.col("cost_return").fill_null(0.0),
        )
        .with_columns(
            (pl.col("gross_return") - pl.col("cost_return")).alias("net_return")
        )
        .select(TIME, "gross_return", "net_return")
        .sort(TIME)
    )


def _proportional_research_costs(
    deltas: pl.DataFrame,
    *,
    config: BacktestConfig,
    slippage_rates: pl.DataFrame | None,
) -> pl.DataFrame:
    """Return cost per unit gross exposure without minimum-fee/account state."""

    if deltas.is_empty():
        return pl.DataFrame(schema={TIME: pl.Date, "cost_return": pl.Float64})
    cost = config.transaction_cost
    trades = _attach_slippage_rates(deltas, slippage_rates).with_columns(
        pl.col("_slippage_rate")
        .fill_null(
            pl.when(pl.col("signed_weight_delta") > 0.0)
            .then(pl.lit(cost.buy_slippage_rate))
            .otherwise(pl.lit(cost.sell_slippage_rate))
        )
        .alias("_effective_slippage")
    )
    return (
        trades.with_columns(
            (
                pl.col("signed_weight_delta").abs()
                * (
                    pl.lit(cost.rate + cost.transfer_fee_rate)
                    + pl.col("_effective_slippage")
                )
                + (-pl.col("signed_weight_delta")).clip(lower_bound=0.0)
                * pl.lit(cost.stamp_tax_rate)
            ).alias("_cost_return")
        )
        .group_by(TIME)
        .agg(pl.col("_cost_return").sum().alias("cost_return"))
        .sort(TIME)
    )


def _daily_turnover_frame(
    requested_weights: pl.DataFrame,
    executed_turnover: pl.DataFrame,
    *,
    requested_deltas: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if requested_weights.is_empty():
        return pl.DataFrame(schema=_daily_turnover_schema())
    requested = (
        _requested_snapshot_turnover(requested_weights)
        if requested_deltas is None
        else requested_deltas.group_by(TIME)
        .agg(pl.col("signed_weight_delta").abs().sum().alias("requested_turnover"))
        .sort(TIME)
    )
    initial_rebalance = (
        requested.get_column(TIME).min() if not requested.is_empty() else None
    )
    return (
        executed_turnover.rename({"turnover": "executed_turnover"})
        .join(requested, on=TIME, how="left")
        .with_columns(
            pl.col("requested_turnover").fill_null(0.0),
            (pl.col(TIME) == pl.lit(initial_rebalance)).alias("is_initial_rebalance"),
        )
        .select(
            TIME,
            "requested_turnover",
            "executed_turnover",
            "is_initial_rebalance",
        )
        .sort(TIME)
    )


def _requested_snapshot_turnover(weights: pl.DataFrame) -> pl.DataFrame:
    if weights.is_empty():
        return pl.DataFrame(schema={TIME: pl.Date, "requested_turnover": pl.Float64})
    return (
        _requested_snapshot_weight_deltas(weights)
        .group_by(TIME)
        .agg(pl.col("signed_weight_delta").abs().sum().alias("requested_turnover"))
        .sort(TIME)
    )


def _requested_snapshot_weight_deltas(weights: pl.DataFrame) -> pl.DataFrame:
    """Return signed changes between complete consecutive target snapshots."""

    if weights.is_empty():
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                ASSET_ID: pl.String,
                "signed_weight_delta": pl.Float64,
            }
        )
    dates = (
        weights.select(TIME)
        .unique()
        .sort(TIME)
        .with_columns(pl.col(TIME).shift(-1).alias("_next_time"))
    )
    current = weights.select(
        TIME,
        ASSET_ID,
        pl.col("weight").alias("signed_weight_delta"),
    )
    previous = (
        weights.join(dates, on=TIME, how="left")
        .drop_nulls("_next_time")
        .select(
            pl.col("_next_time").alias(TIME),
            ASSET_ID,
            (-pl.col("weight")).alias("signed_weight_delta"),
        )
    )
    return (
        pl.concat([current, previous])
        .group_by(TIME, ASSET_ID)
        .agg(pl.col("signed_weight_delta").sum())
        .filter(pl.col("signed_weight_delta") != 0.0)
        .select(TIME, ASSET_ID, "signed_weight_delta")
        .sort([TIME, ASSET_ID])
    )


def _stream_lead_lag_returns(
    book_weights: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    slippage_rates: pl.DataFrame | None,
    sessions: Sequence[date],
    lead_lags: Sequence[int],
    common_dates: Sequence[date],
    progress: DailyDiagnosticProgress | None,
    base_weight_deltas: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if not common_dates:
        return pl.DataFrame(schema=_lead_lag_return_schema())
    common = pl.DataFrame({TIME: common_dates}, schema={TIME: pl.Date})
    frames: list[pl.DataFrame] = []
    total = len(lead_lags)
    for completed, lag in enumerate(lead_lags, start=1):
        weights = _shifted_lead_lag_weights(
            book_weights,
            sessions=sessions,
            lag=int(lag),
            common_dates=common_dates,
        )
        if weights.is_empty():
            return pl.DataFrame(schema=_lead_lag_return_schema())
        result = _research_path_returns(
            weights,
            forward_returns,
            config=config,
            slippage_rates=slippage_rates,
            weight_deltas=(
                None
                if base_weight_deltas is None
                else _shifted_lead_lag_deltas(
                    book_weights,
                    base_weight_deltas,
                    sessions=sessions,
                    lag=int(lag),
                    common_dates=common_dates,
                )
            ),
        )
        frames.append(
            result.join(common, on=TIME, how="inner").select(
                TIME,
                pl.lit(int(lag), dtype=pl.Int64).alias("lag"),
                pl.col("gross_return").cast(pl.Float64),
                pl.col("net_return").cast(pl.Float64),
            )
        )
        if progress is not None:
            progress("book_lead_lag_paths", completed, total)
    return pl.concat(frames).sort(["lag", TIME])


def _stream_named_lead_lag_returns(
    weights_by_path: Mapping[str, pl.DataFrame],
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    slippage_rates: pl.DataFrame | None,
    sessions: Sequence[date],
    lead_lags: Sequence[int],
    common_dates: Sequence[date],
    progress: DailyDiagnosticProgress | None,
    base_weight_deltas: Mapping[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Stream named Book/Tail lag paths over one common date sample."""

    if not common_dates or not weights_by_path:
        return pl.DataFrame(schema=_named_lead_lag_return_schema())
    common = pl.DataFrame({TIME: common_dates}, schema={TIME: pl.Date})
    names = tuple(sorted(weights_by_path))
    frames: list[pl.DataFrame] = []
    total = len(lead_lags) * len(names)
    completed = 0
    for lag in lead_lags:
        shifted = {
            name: _shifted_lead_lag_weights(
                weights_by_path[name],
                sessions=sessions,
                lag=int(lag),
                common_dates=common_dates,
            )
            for name in names
        }
        if any(frame.is_empty() for frame in shifted.values()):
            return pl.DataFrame(schema=_named_lead_lag_return_schema())
        for name in names:
            result = _research_path_returns(
                shifted[name],
                forward_returns,
                config=config,
                slippage_rates=slippage_rates,
                weight_deltas=(
                    None
                    if base_weight_deltas is None
                    else _shifted_lead_lag_deltas(
                        weights_by_path[name],
                        base_weight_deltas[name],
                        sessions=sessions,
                        lag=int(lag),
                        common_dates=common_dates,
                    )
                ),
            )
            frames.append(
                result.join(common, on=TIME, how="inner").select(
                    TIME,
                    pl.lit(name).alias("path_kind"),
                    pl.lit(int(lag), dtype=pl.Int64).alias("lag"),
                    pl.col("gross_return").cast(pl.Float64),
                    pl.col("net_return").cast(pl.Float64),
                )
            )
            completed += 1
        if progress is not None:
            progress("alpha_return_lag_paths", completed, total)
    return pl.concat(frames).sort(["path_kind", "lag", TIME])


def _scheduled_factor(
    signals: ScheduledPrediction,
    *,
    prediction_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if not isinstance(signals, ScheduledPrediction):
        raise TypeError("fixed-window diagnostics require a ScheduledPrediction")
    source = (
        signals.prediction.collect(dense=False)
        if prediction_frame is None
        else prediction_frame
    )
    if "value" not in source.columns and "signal" in source.columns:
        source = source.rename({"signal": "value"})
    values = validate_panel_frame(
        source.rename({"value": "factor"}),
        label="prediction",
        value_columns=("factor",),
    ).rename({TIME: "execution_date"})
    schedule = (
        signals.schedule.drop_nulls("execution_date")
        .select(
            pl.col("rebalance_date").cast(pl.Date).alias("evaluation_date"),
            pl.col("execution_date").cast(pl.Date),
        )
        .unique()
    )
    if schedule.get_column("execution_date").n_unique() != schedule.height:
        raise ValueError("each execution date must map to one evaluation date")
    return (
        values.join(schedule, on="execution_date", how="inner")
        .select("evaluation_date", "execution_date", ASSET_ID, "factor")
        .sort(["evaluation_date", ASSET_ID])
    )


def _validate_scheduled_factor_frame(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"evaluation_date", "execution_date", ASSET_ID, "factor"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"factor is missing required columns: {missing}")
    normalized = (
        frame.select(*required)
        .with_columns(
            pl.col("evaluation_date").cast(pl.Date, strict=False),
            pl.col("execution_date").cast(pl.Date, strict=False),
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("factor").cast(pl.Float64, strict=False),
        )
        .drop_nulls(list(required))
    )
    normalized = normalized.filter(pl.col("factor").is_finite())
    if normalized.select(
        pl.struct("evaluation_date", ASSET_ID).is_duplicated().any()
    ).item():
        raise ValueError("factor must be unique by (evaluation_date, asset_id)")
    return normalized.sort(["evaluation_date", ASSET_ID])


def _validate_windows(windows: Sequence[SessionWindow]) -> tuple[SessionWindow, ...]:
    result = tuple(windows)
    if not result:
        raise ValueError("at least one session window is required")
    if any(not isinstance(window, SessionWindow) for window in result):
        raise TypeError("windows must contain SessionWindow values")
    identifiers = [window.window_id for window in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("window_id values must be unique")
    return result


def _session_dates(calendar: pl.DataFrame | None, prices: pl.DataFrame) -> list[date]:
    source = prices if calendar is None else calendar
    if TIME not in source.columns:
        raise ValueError("calendar is missing required column: time")
    if "is_open" in source.columns:
        source = source.filter(pl.col("is_open").cast(pl.Boolean, strict=False))
    return (
        source.select(pl.col(TIME).cast(pl.Date, strict=False))
        .drop_nulls()
        .unique()
        .sort(TIME)
        .get_column(TIME)
        .to_list()
    )


def _window_schedule(
    schedule: pl.DataFrame,
    sessions: Sequence[date],
    windows: Sequence[SessionWindow],
) -> pl.DataFrame:
    positions = {session: index for index, session in enumerate(sessions)}
    rows: list[dict[str, object]] = []
    for item in schedule.sort("evaluation_date").iter_rows(named=True):
        execution_date = item["execution_date"]
        position = positions.get(execution_date)
        for window in windows:
            start_position = (
                None if position is None else position + window.start_offset
            )
            end_position = None if position is None else position + window.end_offset
            available = (
                start_position is not None
                and end_position is not None
                and end_position < len(sessions)
            )
            rows.append(
                {
                    "evaluation_date": item["evaluation_date"],
                    "execution_date": execution_date,
                    "target_end_date": sessions[end_position] if available else None,
                    "_start_price_date": (
                        sessions[start_position] if available else None
                    ),
                    "window_kind": window.window_kind,
                    "window_id": window.window_id,
                    "start_session": window.start_session,
                    "end_session": window.end_session,
                    "start_offset": window.start_offset,
                    "end_offset": window.end_offset,
                    "target_available": available,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "evaluation_date": pl.Date,
            "execution_date": pl.Date,
            "target_end_date": pl.Date,
            "_start_price_date": pl.Date,
            "window_kind": pl.String,
            "window_id": pl.String,
            "start_session": pl.Int64,
            "end_session": pl.Int64,
            "start_offset": pl.Int64,
            "end_offset": pl.Int64,
            "target_available": pl.Boolean,
        },
    )


def _attach_asof_price(
    rows: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    lookup_column: str,
    output_column: str,
) -> pl.DataFrame:
    lookup = rows.select("_row_id", ASSET_ID, lookup_column).sort(
        [ASSET_ID, lookup_column]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        attached = lookup.join_asof(
            prices,
            left_on=lookup_column,
            right_on="_observed_date",
            by=ASSET_ID,
            strategy="backward",
        ).select("_row_id", pl.col("price").alias(output_column))
    return rows.join(attached, on="_row_id", how="left")


def _quantile_membership(factor: pl.DataFrame, *, quantiles: int) -> pl.DataFrame:
    if factor.is_empty():
        return factor.with_columns(
            pl.lit(None, dtype=pl.String).alias("quantile"),
            pl.lit(0, dtype=pl.Int64).alias("_count"),
            pl.lit(0, dtype=pl.Int64).alias("_unique"),
        )
    lineage = factor.sort(
        ["evaluation_date", "factor", ASSET_ID],
        descending=[False, True, False],
    )
    if not {"_count", "_unique"}.issubset(lineage.columns):
        lineage = lineage.with_columns(
            pl.len().over("evaluation_date").alias("_count"),
            pl.col("factor").n_unique().over("evaluation_date").alias("_unique"),
        )
    return (
        lineage.with_columns(
            pl.int_range(1, pl.len() + 1).over("evaluation_date").alias("_rank")
        )
        .with_columns(
            pl.when(pl.col("_count") >= quantiles)
            .then(
                pl.concat_str(
                    pl.lit("q"),
                    (
                        ((pl.col("_rank") - 1) * quantiles / pl.col("_count")).floor()
                        + 1
                    ).cast(pl.Int64),
                )
            )
            .otherwise(None)
            .alias("quantile")
        )
        .drop("_rank")
    )


def _weighted_window_returns(
    weights: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    weight_column: str,
    return_column: str,
) -> pl.DataFrame:
    required = {"evaluation_date", ASSET_ID, weight_column, "unavailable_reason"}
    missing = sorted(required - set(weights.columns))
    if missing:
        raise ValueError(f"weights are missing required columns: {missing}")
    selected_columns = [*required]
    if "_book_active" in weights.columns:
        selected_columns.append("_book_active")
    paired = forward_returns.join(
        weights.select(*selected_columns),
        on=["evaluation_date", ASSET_ID],
        how="left",
    )
    if paired.is_empty():
        return _empty_window_metric(
            {
                return_column: pl.Float64,
                "expected_count": pl.Int64,
                "observed_count": pl.Int64,
                "coverage_ratio": pl.Float64,
                "unavailable_reason": pl.String,
            }
        )
    active = (
        pl.col("_book_active").fill_null(False)
        if "_book_active" in paired.columns
        else pl.col(weight_column).is_not_null() & (pl.col(weight_column) != 0.0)
    )
    result = (
        paired.group_by(*_RETURN_GROUP_COLUMNS)
        .agg(
            active.sum().alias("expected_count"),
            (active & pl.col("forward_return").is_not_null())
            .sum()
            .alias("observed_count"),
            pl.when(active & pl.col("forward_return").is_not_null())
            .then(pl.col(weight_column) * pl.col("forward_return"))
            .otherwise(0.0)
            .sum()
            .alias("_weighted_return"),
            pl.col("unavailable_reason").drop_nulls().first().alias("_weight_reason"),
        )
        .with_columns(
            pl.when(pl.col("expected_count") > 0)
            .then(pl.col("observed_count") / pl.col("expected_count"))
            .otherwise(None)
            .alias("coverage_ratio")
        )
        .with_columns(
            pl.when(pl.col("_weight_reason").is_null() & (pl.col("expected_count") > 0))
            .then(pl.col("_weighted_return"))
            .otherwise(None)
            .alias(return_column),
            pl.when(pl.col("_weight_reason").is_not_null())
            .then(pl.col("_weight_reason"))
            .when(pl.col("expected_count") == 0)
            .then(pl.lit("weights are unavailable"))
            .otherwise(None)
            .alias("unavailable_reason"),
        )
        .select(
            *_RETURN_GROUP_COLUMNS,
            return_column,
            "expected_count",
            "observed_count",
            "coverage_ratio",
            "unavailable_reason",
        )
        .sort(["evaluation_date", "window_id"])
    )
    return result


def _safe_corr(left: str, right: str) -> pl.Expr:
    return (
        pl.when(
            (pl.len() < 2)
            | (pl.col(left).n_unique() < 2)
            | (pl.col(right).n_unique() < 2)
        )
        .then(None)
        .otherwise(pl.corr(left, right))
    )


def summarize_window_ic(
    ic: pl.DataFrame,
    *,
    annualization_sessions: int,
) -> pl.DataFrame:
    schema = {
        "method": pl.String,
        "window_kind": pl.String,
        "window_id": pl.String,
        "start_session": pl.Int64,
        "end_session": pl.Int64,
        "mean": pl.Float64,
        "standard_deviation": pl.Float64,
        "positive_ratio": pl.Float64,
        "icir": pl.Float64,
        "annualization_frequency": pl.Float64,
        "sample_size": pl.Int64,
    }
    rows: list[dict[str, object]] = []
    for method, column in (("pearson", "pearson_ic"), ("spearman", "spearman_ic")):
        for key, sample in ic.group_by(*_WINDOW_COLUMNS):
            metadata = dict(zip(_WINDOW_COLUMNS, key, strict=True))
            values = np.asarray(
                [float(v) for v in sample.get_column(column) if v is not None],
                dtype=float,
            )
            width = int(metadata["end_session"]) - int(metadata["start_session"]) + 1
            frequency = annualization_sessions / width
            mean = float(values.mean()) if values.size else None
            deviation = float(values.std(ddof=1)) if values.size >= 2 else None
            rows.append(
                {
                    "method": method,
                    **metadata,
                    "mean": mean,
                    "standard_deviation": deviation,
                    "positive_ratio": (
                        float(np.mean(values > 0.0)) if values.size else None
                    ),
                    "icir": (
                        None
                        if deviation in {None, 0.0}
                        else float(mean) / float(deviation) * math.sqrt(frequency)
                    ),
                    "annualization_frequency": frequency,
                    "sample_size": int(values.size),
                }
            )
    return (
        pl.DataFrame(rows, schema=schema).sort(["method", "window_kind", "end_session"])
        if rows
        else pl.DataFrame(schema=schema)
    )


def summarize_signal_persistence(
    series: pl.DataFrame,
    horizons: Sequence[int],
) -> pl.DataFrame:
    schema = {
        "horizon_sessions": pl.Int64,
        "mean_rank_autocorrelation": pl.Float64,
        "sample_size": pl.Int64,
        "signal_half_life_band": pl.String,
    }
    means = (
        series.group_by("horizon_sessions").agg(
            pl.col("rank_autocorrelation").mean().alias("mean_rank_autocorrelation"),
            pl.col("rank_autocorrelation").is_not_null().sum().alias("sample_size"),
        )
        if not series.is_empty()
        else pl.DataFrame(
            schema={
                "horizon_sessions": pl.Int64,
                "mean_rank_autocorrelation": pl.Float64,
                "sample_size": pl.Int64,
            }
        )
    )
    grid = (
        pl.DataFrame(
            {"horizon_sessions": [int(value) for value in horizons]},
            schema={"horizon_sessions": pl.Int64},
        )
        .join(means, on="horizon_sessions", how="left")
        .with_columns(pl.col("sample_size").fill_null(0).cast(pl.Int64))
        .sort("horizon_sessions")
    )
    band = "unavailable"
    previous = 0
    observed = grid.drop_nulls("mean_rank_autocorrelation")
    if not observed.is_empty():
        band = ">120D"
        for row in observed.iter_rows(named=True):
            horizon = int(row["horizon_sessions"])
            if float(row["mean_rank_autocorrelation"]) <= 0.5:
                band = f"{previous}-{horizon}D"
                break
            previous = horizon
        else:
            if previous < max(horizons):
                band = "unavailable"
    return grid.with_columns(pl.lit(band).alias("signal_half_life_band")).select(
        *schema
    )


def _empty_forward_returns() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "evaluation_date": pl.Date,
            "execution_date": pl.Date,
            "target_end_date": pl.Date,
            ASSET_ID: pl.String,
            "window_kind": pl.String,
            "window_id": pl.String,
            "start_session": pl.Int64,
            "end_session": pl.Int64,
            "start_offset": pl.Int64,
            "end_offset": pl.Int64,
            "forward_return": pl.Float64,
        }
    )


def _empty_weight_frame(column: str) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "evaluation_date": pl.Date,
            "execution_date": pl.Date,
            ASSET_ID: pl.String,
            column: pl.Float64,
            "unavailable_reason": pl.String,
        }
    )


def _empty_window_metric(extra: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "evaluation_date": pl.Date,
            "execution_date": pl.Date,
            "target_end_date": pl.Date,
            "window_kind": pl.String,
            "window_id": pl.String,
            "start_session": pl.Int64,
            "end_session": pl.Int64,
            **extra,
        }
    )


def _daily_return_schema() -> dict[str, pl.DataType]:
    return {
        TIME: pl.Date,
        "gross_return": pl.Float64,
        "net_return": pl.Float64,
    }


def _daily_quantile_return_schema() -> dict[str, pl.DataType]:
    return {
        TIME: pl.Date,
        "quantile": pl.String,
        "gross_return": pl.Float64,
        "constituent_count": pl.Int64,
        "unavailable_reason": pl.String,
    }


def _daily_turnover_schema() -> dict[str, pl.DataType]:
    return {
        TIME: pl.Date,
        "requested_turnover": pl.Float64,
        "executed_turnover": pl.Float64,
        "is_initial_rebalance": pl.Boolean,
    }


def _lead_lag_return_schema() -> dict[str, pl.DataType]:
    return {
        TIME: pl.Date,
        "lag": pl.Int64,
        "gross_return": pl.Float64,
        "net_return": pl.Float64,
    }


def _named_lead_lag_return_schema() -> dict[str, pl.DataType]:
    return {
        TIME: pl.Date,
        "path_kind": pl.String,
        "lag": pl.Int64,
        "gross_return": pl.Float64,
        "net_return": pl.Float64,
    }


def _rolling_ic_schema() -> dict[str, pl.DataType]:
    return {
        "evaluation_date": pl.Date,
        "window_kind": pl.String,
        "window_id": pl.String,
        "start_session": pl.Int64,
        "end_session": pl.Int64,
        "method": pl.String,
        "rolling_ic": pl.Float64,
        "rolling_observations": pl.Int64,
    }


def _inference_schema() -> dict[str, pl.DataType]:
    return {
        "metric": pl.String,
        "window_kind": pl.String,
        "window_id": pl.String,
        "start_session": pl.Int64,
        "end_session": pl.Int64,
        "window_width": pl.Int64,
        "mean": pl.Float64,
        "hac_standard_error": pl.Float64,
        "hac_t": pl.Float64,
        "p_value": pl.Float64,
        "q_value": pl.Float64,
        "confidence_low": pl.Float64,
        "confidence_high": pl.Float64,
        "sample_size": pl.Int64,
        "hac_lag": pl.Int64,
        "unavailable_reason": pl.String,
        "cohort_count": pl.Int64,
        "cohort_mean_median": pl.Float64,
        "cohort_mean_min": pl.Float64,
        "cohort_mean_max": pl.Float64,
        "cohort_same_sign_ratio": pl.Float64,
    }


__all__ = [
    "DAILY_ALPHA_RETURN_LAGS",
    "DAILY_BOOK_LEAD_LAGS",
    "DAILY_BUCKET_WINDOWS",
    "DAILY_CUMULATIVE_WINDOWS",
    "DAILY_ROLLING_IC_OBSERVATIONS",
    "DAILY_SESSION_WINDOWS",
    "DAILY_SUMMARY_AUTOCORRELATION_LAGS",
    "SIGNAL_PERSISTENCE_HORIZONS",
    "DailyRankPathDiagnostics",
    "HACMeanTest",
    "PredictionHorizonDiagnostics",
    "SessionWindow",
    "benjamini_hochberg",
    "build_statistical_inference",
    "centered_rank_book_weights",
    "gross_one_tail_weights",
    "hac_mean_test",
    "implied_signal_half_life",
    "non_overlapping_cohort_statistics",
    "quantile_curve_structure",
    "rolling_window_information_coefficients",
    "run_daily_rank_path_diagnostics",
    "run_prediction_horizon_diagnostics",
    "session_window_forward_returns",
    "signal_rank_persistence",
    "summarize_signal_persistence",
    "summarize_window_ic",
    "window_book_returns",
    "window_factor_returns",
    "window_information_coefficients",
    "window_quantile_forward_returns",
    "window_tail_returns",
]
