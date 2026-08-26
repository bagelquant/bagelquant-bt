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
from dataclasses import dataclass
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
    book_turnover: pl.DataFrame
    book_lead_lag_returns: pl.DataFrame
    alpha_return_lag_returns: pl.DataFrame
    signal_autocorrelation: pl.DataFrame
    rolling_ic: pl.DataFrame


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
    if factor.is_empty() or available.is_empty():
        return _empty_forward_returns(), coverage.with_columns(
            pl.lit(0, dtype=pl.Int64).alias("observed_count"),
            pl.lit(None, dtype=pl.Float64).alias("coverage_ratio"),
        )

    rows = (
        factor.join(
            available,
            on=["evaluation_date", "execution_date"],
            how="inner",
        )
        .with_row_index("_row_id")
    )
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
    price_lookup = market.select(
        ASSET_ID,
        pl.col(TIME).alias("_observed_date"),
        "price",
    ).sort([ASSET_ID, "_observed_date"])
    rows = _attach_asof_price(
        rows,
        price_lookup,
        lookup_column="_start_price_date",
        output_column="_start_price",
    )
    rows = _attach_asof_price(
        rows,
        price_lookup,
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
        )
        .sort(["evaluation_date", "window_id", ASSET_ID])
    )
    observed = (
        forward.group_by(*_RETURN_GROUP_COLUMNS)
        .agg(pl.col("forward_return").is_not_null().sum().alias("observed_count"))
    )
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


def centered_rank_book_weights(factor: pl.DataFrame) -> pl.DataFrame:
    """Create centered average-rank weights with net zero and gross one."""

    normalized = _validate_scheduled_factor_frame(factor)
    if normalized.is_empty():
        return _empty_weight_frame("book_weight")
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
    return (
        ranked.with_columns(
            pl.when(valid)
            .then(pl.col("_centered") / pl.col("_gross"))
            .otherwise(None)
            .alias("book_weight"),
            pl.when(valid)
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.lit("book requires at least two non-constant ranks"))
            .alias("unavailable_reason"),
        )
        .select(
            "evaluation_date",
            "execution_date",
            ASSET_ID,
            "book_weight",
            "unavailable_reason",
        )
        .sort(["evaluation_date", ASSET_ID])
    )


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
                pl.lit(
                    f"tail requires {quantiles} assets and a non-constant signal"
                )
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
        pl.col("forward_return").is_not_null().sum().over(*_RETURN_GROUP_COLUMNS)
        .cast(pl.Int64)
        .alias("sample_size"),
        pl.col("factor").rank("average").over(*_RETURN_GROUP_COLUMNS).alias(
            "_factor_rank"
        ),
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
    membership: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    """Aggregate one label window with a caller-reused quantile membership."""

    paired = forward_returns.join(
        membership.select(
            "evaluation_date", ASSET_ID, "quantile", "_count", "_unique"
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
    """Summarize monotonicity and quantile-rank IC for each curve."""

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
        reason = None
        if complete:
            signal_order = np.arange(quantiles, 0, -1, dtype=float)
            return_ranks = stats.rankdata(
                np.asarray(values, dtype=float), method="average"
            )
            if np.unique(return_ranks).size >= 2:
                quantile_ic = float(
                    np.corrcoef(signal_order, return_ranks)[0, 1]
                )
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
                "unavailable_reason": reason,
            }
        )
    return pl.DataFrame(rows, schema=schema).sort(
        ["evaluation_date", "window_id"]
    ) if rows else pl.DataFrame(schema=schema)


def window_factor_returns(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Estimate per-date cross-sectional OLS slopes for every window."""

    normalized = _validate_scheduled_factor_frame(factor)
    paired = forward_returns.join(
        normalized.select("evaluation_date", ASSET_ID, "factor"),
        on=["evaluation_date", ASSET_ID],
        how="left",
    ).drop_nulls(["factor", "forward_return"])
    if paired.is_empty():
        return _empty_window_metric(
            {"factor_return": pl.Float64, "sample_size": pl.Int64}
        )
    return (
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


def signal_rank_persistence(
    factor: pl.DataFrame,
    *,
    calendar: pl.DataFrame,
    horizons: Sequence[int] = SIGNAL_PERSISTENCE_HORIZONS,
    progress: DailyDiagnosticProgress | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Calculate same-universe rank autocorrelation and a grid half-life band."""

    normalized = _validate_scheduled_factor_frame(factor)
    sessions = _session_dates(calendar, normalized.rename({"factor": "price"}).select(
        pl.col("evaluation_date").alias(TIME), ASSET_ID, "price"
    ))
    dates = normalized.get_column("evaluation_date").unique().sort().to_list()
    positions = {session: index for index, session in enumerate(sessions)}
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
    current = normalized.select("evaluation_date", ASSET_ID, "factor")
    future = normalized.select(
        pl.col("evaluation_date").alias("target_date"),
        ASSET_ID,
        pl.col("factor").alias("future_factor"),
    )
    series_frames: list[pl.DataFrame] = []
    total = len(resolved_horizons)
    for completed, horizon in enumerate(resolved_horizons, start=1):
        mappings = [
            {
                "evaluation_date": evaluation_date,
                "target_date": sessions[target_position],
                "horizon_sessions": horizon,
            }
            for evaluation_date in dates
            if (position := positions.get(evaluation_date)) is not None
            and (target_position := position + horizon) < len(sessions)
        ]
        mapping = pl.DataFrame(
            mappings,
            schema={
                "evaluation_date": pl.Date,
                "target_date": pl.Date,
                "horizon_sessions": pl.Int64,
            },
        )
        if not mapping.is_empty():
            paired = (
                mapping.join(current, on="evaluation_date", how="inner")
                .join(future, on=["target_date", ASSET_ID], how="inner")
                .with_columns(
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
            mean, None, None, None, None, None, sample_size, lag,
            "at least two samples required",
        )
    if lag >= sample_size:
        return HACMeanTest(
            mean, None, None, None, None, None, sample_size, lag,
            "sample size must exceed the HAC lag",
        )
    centered = finite - float(mean)
    long_run_variance = float(np.dot(centered, centered) / sample_size)
    for offset in range(1, lag + 1):
        covariance = float(
            np.dot(centered[offset:], centered[:-offset]) / sample_size
        )
        bartlett_weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * covariance
    if not math.isfinite(long_run_variance) or long_run_variance <= 0.0:
        return HACMeanTest(
            mean, None, None, None, None, None, sample_size, lag,
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
            np.mean([math.copysign(1.0, value) == math.copysign(1.0, overall_mean)
                     for value in means if value != 0.0])
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


def run_prediction_horizon_diagnostics(
    signals: ScheduledPrediction,
    prices: pl.DataFrame,
    *,
    windows: Sequence[SessionWindow] = DAILY_SESSION_WINDOWS,
    calendar: pl.DataFrame | None = None,
    quantiles: int = 10,
    annualization_sessions: int = 240,
) -> PredictionHorizonDiagnostics:
    """Run fixed-window diagnostics while retaining one label window at a time."""

    if annualization_sessions <= 0:
        raise ValueError("annualization_sessions must be positive")
    factor = _scheduled_factor(signals)
    market = validate_prices(prices)
    resolved_windows = _validate_windows(windows)
    resolved_calendar = (
        calendar if calendar is not None else market.select(TIME).unique()
    )
    sessions = _session_dates(resolved_calendar, market)
    book_weights = centered_rank_book_weights(factor)
    tail_weights = gross_one_tail_weights(factor, quantiles=quantiles)
    quantile_membership = _quantile_membership(factor, quantiles=quantiles)
    coverage_frames: list[pl.DataFrame] = []
    ic_frames: list[pl.DataFrame] = []
    book_return_frames: list[pl.DataFrame] = []
    tail_return_frames: list[pl.DataFrame] = []
    quantile_return_frames: list[pl.DataFrame] = []
    factor_return_frames: list[pl.DataFrame] = []
    max_window_forward_rows = 0
    for window in resolved_windows:
        forward, window_coverage = _session_window_forward_returns_from_frames(
            factor,
            market,
            windows=(window,),
            sessions=sessions,
        )
        max_window_forward_rows = max(max_window_forward_rows, forward.height)
        coverage_frames.append(window_coverage)
        ic_frames.append(window_information_coefficients(factor, forward))
        book_return_frames.append(window_book_returns(book_weights, forward))
        tail_return_frames.append(window_tail_returns(tail_weights, forward))
        quantile_return_frames.append(
            _window_quantile_forward_returns_with_membership(
                quantile_membership,
                forward,
                quantiles=quantiles,
            )
        )
        factor_return_frames.append(window_factor_returns(factor, forward))
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
    quantile_returns = pl.concat(
        quantile_return_frames, how="diagonal_relaxed"
    ).sort(["evaluation_date", "window_id", "quantile"])
    factor_returns = pl.concat(
        factor_return_frames, how="diagonal_relaxed"
    ).sort(["evaluation_date", "window_id"])
    structure = quantile_curve_structure(quantile_returns, quantiles=quantiles)
    persistence, persistence_summary = signal_rank_persistence(
        factor, calendar=resolved_calendar
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
    schema = _rolling_ic_schema()
    rows: list[dict[str, object]] = []
    for key, sample in ic.group_by(*_WINDOW_COLUMNS):
        metadata = dict(zip(_WINDOW_COLUMNS, key, strict=True))
        ordered = sample.sort("evaluation_date")
        for method, column in (
            ("pearson", "pearson_ic"),
            ("spearman", "spearman_ic"),
        ):
            valid_values: list[float] = []
            for evaluation_date, value in ordered.select(
                "evaluation_date", column
            ).iter_rows():
                rolling_value = None
                if value is not None and math.isfinite(float(value)):
                    valid_values.append(float(value))
                    if len(valid_values) >= observations:
                        rolling_value = float(
                            np.mean(valid_values[-observations:])
                        )
                rows.append(
                    {
                        "evaluation_date": evaluation_date,
                        **metadata,
                        "method": method,
                        "rolling_ic": rolling_value,
                        "rolling_observations": min(
                            len(valid_values), observations
                        ),
                    }
                )
    return (
        pl.DataFrame(rows, schema=schema).sort(
            ["window_kind", "end_session", "method", "evaluation_date"]
        )
        if rows
        else pl.DataFrame(schema=schema)
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
    market = validate_prices(prices)
    factor = _scheduled_factor(signals)
    book = centered_rank_book_weights(factor)
    tail = gross_one_tail_weights(factor, quantiles=quantiles)
    book_weights = _path_weight_frame(book, column="book_weight")
    tail_weights = _path_weight_frame(tail, column="tail_weight")
    market_sessions = _session_dates(None, market)
    common_lead_lag_dates = _common_lead_lag_dates(
        book_weights,
        sessions=market_sessions,
        lead_lags=resolved_lags,
    )
    prepared = _prepare_price_data(market, inputs_sorted=True)
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
    book_daily_returns = _research_path_returns(
        book_weights,
        prepared.forward_returns,
        config=config,
        slippage_rates=resolved_slippage,
    )
    tail_daily_returns = _research_path_returns(
        tail_weights,
        prepared.forward_returns,
        config=config,
        slippage_rates=resolved_slippage,
    )
    executed_turnover = _sparse_executed_turnover(
        book_weights,
        market,
        prepared.forward_returns,
        execution_availability=resolved_availability,
        retry_blocked=config.retry_blocked_orders,
    )
    if progress is not None:
        progress("book_tail_paths", 1, 1)
    book_turnover = _daily_turnover_frame(book_weights, executed_turnover)
    lead_lag_returns = _stream_lead_lag_returns(
        book_weights,
        prepared.forward_returns,
        config=config,
        slippage_rates=resolved_slippage,
        sessions=market_sessions,
        lead_lags=resolved_lags,
        common_dates=common_lead_lag_dates,
        progress=progress,
    )
    alpha_return_common_dates = _common_named_lead_lag_dates(
        base_weights,
        sessions=market_sessions,
        lead_lags=resolved_alpha_return_lags,
    )
    alpha_return_lag_returns = _stream_named_lead_lag_returns(
        base_weights,
        prepared.forward_returns,
        config=config,
        slippage_rates=resolved_slippage,
        sessions=market_sessions,
        lead_lags=resolved_alpha_return_lags,
        common_dates=alpha_return_common_dates,
        progress=progress,
    )

    resolved_calendar = (
        calendar if calendar is not None else market.select(TIME).unique()
    )
    autocorrelation, _ = signal_rank_persistence(
        factor,
        calendar=resolved_calendar,
        horizons=resolved_autocorrelation_lags,
        progress=progress,
    )
    resolved_ic = horizon_ic
    if resolved_ic is None:
        sessions = _session_dates(resolved_calendar, market)
        ic_frames = []
        for window in DAILY_SESSION_WINDOWS:
            forward, _ = _session_window_forward_returns_from_frames(
                factor,
                market,
                windows=(window,),
                sessions=sessions,
            )
            ic_frames.append(window_information_coefficients(factor, forward))
        resolved_ic = pl.concat(
            ic_frames,
            how="diagonal_relaxed",
        ).sort(["evaluation_date", "window_id"])
    return DailyRankPathDiagnostics(
        book_daily_returns=book_daily_returns,
        tail_daily_returns=tail_daily_returns,
        book_turnover=book_turnover,
        book_lead_lag_returns=lead_lag_returns,
        alpha_return_lag_returns=alpha_return_lag_returns,
        signal_autocorrelation=autocorrelation,
        rolling_ic=rolling_window_information_coefficients(
            resolved_ic,
            observations=rolling_observations,
        ),
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
        .sort([TIME, ASSET_ID])
    )


def _research_path_returns(
    weights: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    slippage_rates: pl.DataFrame | None,
) -> pl.DataFrame:
    """Calculate a capital-free factor path with proportional trading costs."""

    if weights.is_empty():
        return pl.DataFrame(schema=_daily_return_schema())
    timeline = weights.select(TIME).unique().sort(TIME)
    gross = (
        weights.join(forward_returns, on=[TIME, ASSET_ID], how="left")
        .with_columns(
            (
                pl.col("weight")
                * pl.col("forward_return").fill_null(0.0)
            ).alias("_weighted_return")
        )
        .group_by(TIME)
        .agg(pl.col("_weighted_return").sum().alias("gross_return"))
    )
    deltas = _requested_snapshot_weight_deltas(weights)
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
            (pl.col("gross_return") - pl.col("cost_return")).alias(
                "net_return"
            )
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
                pl.col("weight_delta")
                * (
                    pl.lit(cost.rate + cost.transfer_fee_rate)
                    + pl.col("_effective_slippage")
                )
                + (-pl.col("signed_weight_delta"))
                .clip(lower_bound=0.0)
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
) -> pl.DataFrame:
    if requested_weights.is_empty():
        return pl.DataFrame(schema=_daily_turnover_schema())
    requested = _requested_snapshot_turnover(requested_weights)
    initial_rebalance = (
        requested.get_column(TIME).min() if not requested.is_empty() else None
    )
    return (
        executed_turnover.rename({"turnover": "executed_turnover"})
        .join(requested, on=TIME, how="left")
        .with_columns(
            pl.col("requested_turnover").fill_null(0.0),
            (pl.col(TIME) == pl.lit(initial_rebalance)).alias(
                "is_initial_rebalance"
            ),
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
        return pl.DataFrame(
            schema={TIME: pl.Date, "requested_turnover": pl.Float64}
        )
    return (
        _requested_snapshot_weight_deltas(weights)
        .group_by(TIME)
        .agg(pl.col("weight_delta").sum().alias("requested_turnover"))
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
                "weight_delta": pl.Float64,
            }
        )
    dates = (
        weights.select(TIME)
        .unique()
        .sort(TIME)
        .with_columns(pl.col(TIME).shift(1).alias("_previous_time"))
    )
    current_keys = weights.select(TIME, ASSET_ID)
    previous_keys = (
        weights.select(pl.col(TIME).alias("_previous_time"), ASSET_ID)
        .join(dates, on="_previous_time", how="inner")
        .select(TIME, ASSET_ID)
    )
    keys = pl.concat([current_keys, previous_keys]).unique()
    previous = weights.select(
        pl.col(TIME).alias("_previous_time"),
        ASSET_ID,
        pl.col("weight").alias("_previous_weight"),
    )
    return (
        keys.join(dates, on=TIME, how="left")
        .join(weights, on=[TIME, ASSET_ID], how="left")
        .join(previous, on=["_previous_time", ASSET_ID], how="left")
        .with_columns(
            (
                pl.col("weight").fill_null(0.0)
                - pl.col("_previous_weight").fill_null(0.0)
            ).alias("signed_weight_delta")
        )
        .with_columns(
            pl.col("signed_weight_delta").abs().alias("weight_delta")
        )
        .filter(pl.col("weight_delta") != 0.0)
        .select(TIME, ASSET_ID, "signed_weight_delta", "weight_delta")
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
            )
            frames.append(
                result.join(common, on=TIME, how="inner")
                .select(
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


def _scheduled_factor(signals: ScheduledPrediction) -> pl.DataFrame:
    if not isinstance(signals, ScheduledPrediction):
        raise TypeError("fixed-window diagnostics require a ScheduledPrediction")
    values = validate_panel_frame(
        signals.prediction.collect(dense=False).rename({"value": "factor"}),
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
    normalized = frame.select(*required).with_columns(
        pl.col("evaluation_date").cast(pl.Date, strict=False),
        pl.col("execution_date").cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("factor").cast(pl.Float64, strict=False),
    ).drop_nulls(list(required))
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
        attached = (
            lookup.join_asof(
                prices,
                left_on=lookup_column,
                right_on="_observed_date",
                by=ASSET_ID,
                strategy="backward",
            )
            .select("_row_id", pl.col("price").alias(output_column))
        )
    return rows.join(attached, on="_row_id", how="left")


def _quantile_membership(factor: pl.DataFrame, *, quantiles: int) -> pl.DataFrame:
    if factor.is_empty():
        return factor.with_columns(
            pl.lit(None, dtype=pl.String).alias("quantile"),
            pl.lit(0, dtype=pl.Int64).alias("_count"),
            pl.lit(0, dtype=pl.Int64).alias("_unique"),
        )
    return (
        factor.sort(
            ["evaluation_date", "factor", ASSET_ID],
            descending=[False, True, False],
        )
        .with_columns(
            pl.len().over("evaluation_date").alias("_count"),
            pl.col("factor").n_unique().over("evaluation_date").alias("_unique"),
            pl.int_range(1, pl.len() + 1).over("evaluation_date").alias("_rank"),
        )
        .with_columns(
            pl.when(pl.col("_count") >= quantiles)
            .then(
                pl.concat_str(
                    pl.lit("q"),
                    (
                        ((pl.col("_rank") - 1) * quantiles / pl.col("_count"))
                        .floor()
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
    paired = forward_returns.join(
        weights.select(*required),
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
    active = pl.col(weight_column).is_not_null() & (pl.col(weight_column) != 0.0)
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
            pl.when(
                pl.col("_weight_reason").is_null()
                & (pl.col("expected_count") > 0)
                & (pl.col("observed_count") == pl.col("expected_count"))
            )
            .then(pl.col("_weighted_return"))
            .otherwise(None)
            .alias(return_column),
            pl.when(pl.col("_weight_reason").is_not_null())
            .then(pl.col("_weight_reason"))
            .when(pl.col("expected_count") == 0)
            .then(pl.lit("weights are unavailable"))
            .when(pl.col("observed_count") != pl.col("expected_count"))
            .then(pl.lit("forward-return coverage is incomplete"))
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
    return pl.DataFrame(rows, schema=schema).sort(
        ["method", "window_kind", "end_session"]
    ) if rows else pl.DataFrame(schema=schema)


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
        series.group_by("horizon_sessions")
        .agg(
            pl.col("rank_autocorrelation").mean().alias(
                "mean_rank_autocorrelation"
            ),
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
    grid = pl.DataFrame(
        {"horizon_sessions": [int(value) for value in horizons]},
        schema={"horizon_sessions": pl.Int64},
    ).join(means, on="horizon_sessions", how="left").with_columns(
        pl.col("sample_size").fill_null(0).cast(pl.Int64)
    ).sort("horizon_sessions")
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
