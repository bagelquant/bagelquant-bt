"""Pure rank, label-window and inference primitives for saved evaluations."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import polars as pl

from .inputs import (
    ASSET_ID,
    TIME,
)

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

    Forward labels are supplied by the caller. The complete result retains
    aggregate observations so callers never need every asset/window label
    resident at once.
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
            from scipy.stats import rankdata

            return_ranks = rankdata(return_values, method="average")
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

    prepared = _prepare_factor_returns(factor, standardization)
    return _window_factor_returns_prepared(prepared, forward_returns,
        complete=standardization == "cross_sectional_zscore")


def _prepare_factor_returns(factor, standardization):
    if standardization not in {"none", "cross_sectional_zscore"}:
        raise ValueError(f"unsupported factor standardization: {standardization}")

    normalized = _validate_scheduled_factor_frame(factor)
    if standardization == "cross_sectional_zscore":
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
    return normalized


def _window_factor_returns_prepared(normalized, forward_returns, *, complete):
    """One numerical formula consuming shared, validated cross sections."""
    window_groups = (
        forward_returns.select(*_RETURN_GROUP_COLUMNS).unique() if complete else None
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
    from scipy import stats

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
    """Summarize raw ICIR and its separately named annualized ratio."""

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
        "annualized_icir": pl.Float64,
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
                        else float(mean) / float(deviation)
                    ),
                    "annualized_icir": (
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
    "signal_rank_persistence",
    "summarize_signal_persistence",
    "summarize_window_ic",
    "window_book_returns",
    "window_factor_returns",
    "window_information_coefficients",
    "window_quantile_forward_returns",
    "window_tail_returns",
]
