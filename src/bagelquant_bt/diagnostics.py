"""Pure saved-value diagnostics with caller-selected periods and factor names."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date

import polars as pl

from .horizon import PredictionHorizonDiagnostics


def annualized_ratio(
    frame: pl.DataFrame, column: str, annualization: int
) -> float | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    values = frame.get_column(column).drop_nulls()
    values = values.filter(values.is_finite())
    if values.len() < 2:
        return None
    mean, std = values.mean(), values.std(ddof=1)
    if mean is None or std is None or std == 0:
        return None
    value = float(mean) / float(std) * math.sqrt(annualization)
    return value if math.isfinite(value) else None


def signal_coverage(
    signal_frame: pl.DataFrame,
    selected_universe: pl.DataFrame,
) -> pl.DataFrame:
    """Count finite daily signals and selected-Universe members."""

    required = {"time", "asset_id"}
    missing = sorted(required - set(selected_universe.columns))
    if missing:
        raise ValueError(f"selected Universe is missing required columns: {missing}")
    universe = selected_universe
    if "active" in universe.columns:
        universe = universe.filter(pl.col("active").fill_null(False))
    elif "eligible" in universe.columns:
        universe = universe.filter(pl.col("eligible").fill_null(False))
    universe = (
        universe.select("time", "asset_id")
        .with_columns(
            pl.col("time").cast(pl.Date),
            pl.col("asset_id").cast(pl.String),
        )
        .unique()
    )
    universe_counts = universe.group_by("time").agg(pl.len().alias("universe_count"))
    source = signal_frame
    value_column = "value" if "value" in source.columns else "signal"
    signal = (
        source.select(
            "time",
            "asset_id",
            pl.col(value_column).cast(pl.Float64).alias("value"),
        )
        .filter(pl.col("value").is_finite())
        .join(universe, on=["time", "asset_id"], how="inner")
        .group_by("time")
        .agg(pl.len().alias("signal_count"))
    )
    return (
        universe_counts.join(signal, on="time", how="left")
        .with_columns(
            pl.col("signal_count").fill_null(0).cast(pl.Int64),
            pl.col("universe_count").cast(pl.Int64),
        )
        .select("time", "signal_count", "universe_count")
        .sort("time")
    )


def stability_tables(
    diagnostics: PredictionHorizonDiagnostics,
    *,
    periods: Mapping[str, tuple[date, date]],
    rolling_observations: int = 240,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize only labels known by each requested segment's ending.

    Nonempty economic observations require explicit ``target_end_date`` and
    ``available_date``. Rolling means retain the maximum economic/knowledge
    dates of their actual contributors for subsequent point-in-time slicing.
    """

    if (
        isinstance(rolling_observations, bool)
        or not isinstance(rolling_observations, int)
        or rolling_observations < 1
    ):
        raise ValueError("rolling observations must be positive")
    if any(beginning > ending for beginning, ending in periods.values()):
        raise ValueError("stability period beginning must not follow ending")
    sources = (
        (diagnostics.ic, "pearson_ic", "pearson_ic"),
        (diagnostics.ic, "spearman_ic", "spearman_ic"),
        (diagnostics.book_returns, "book_return", "book_return"),
        (diagnostics.tail_returns, "tail_return", "tail_return"),
        (
            diagnostics.quantile_structure,
            "quantile_rank_ic",
            "quantile_rank_ic",
        ),
        (diagnostics.factor_returns, "factor_return", "cross_section_regression"),
    )
    segment_rows: list[dict[str, object]] = []
    rolling_frames: list[pl.DataFrame] = []
    groups = ("window_kind", "window_id", "start_session", "end_session")
    dates = ("target_start_date", "target_end_date", "available_date")
    for frame, column, metric in sources:
        if frame.is_empty() or column not in frame.columns:
            continue
        if not {"target_end_date", "available_date"} <= set(frame.columns):
            raise ValueError(
                "stability economic observations require target_end_date "
                "and available_date"
            )
        frame = frame.with_columns(
            pl.col("target_end_date", "available_date").cast(pl.Date),
            (
                pl.col("target_start_date").cast(pl.Date)
                if "target_start_date" in frame.columns
                else pl.lit(None, dtype=pl.Date)
            ).alias("target_start_date"),
        )
        if frame.select(
            pl.any_horizontal(
                pl.col("target_end_date").is_null(),
                pl.col("available_date").is_null(),
            ).any()
        ).item():
            raise ValueError("stability economic/knowledge dates must be nonnull")
        selected = frame.select(
            "evaluation_date",
            *groups,
            *dates,
            pl.col(column).alias("value"),
        )
        for segment, (beginning, ending) in periods.items():
            predicate = (
                pl.col("evaluation_date").is_between(beginning, ending)
                & (pl.col("target_end_date") <= ending)
                & (pl.col("available_date") <= ending)
            )
            sample = selected.filter(predicate)
            for key, _ in selected.group_by(*groups):
                values = sample.filter(
                    pl.all_horizontal(
                        pl.col(name) == value
                        for name, value in zip(groups, key, strict=True)
                    )
                )
                finite = values.filter(
                    pl.col("value").is_not_null() & pl.col("value").is_finite()
                )
                mean = finite.get_column("value").mean()
                reason = (
                    "no mature observations in requested segment"
                    if values.is_empty()
                    else "no finite observations in requested segment"
                    if finite.is_empty()
                    else "segment mean is non-finite"
                    if mean is None or not math.isfinite(mean)
                    else None
                )
                segment_rows.append(
                    {
                        "segment": segment,
                        "metric": metric,
                        "window_kind": key[0],
                        "window_id": key[1],
                        "start_session": key[2],
                        "end_session": key[3],
                        "mean": mean if reason is None else None,
                        "sample_size": finite.height,
                        "status": "unavailable" if reason else "complete",
                        "reason": reason,
                        "start_date": (
                            None
                            if finite.is_empty()
                            else finite.get_column("evaluation_date").min()
                        ),
                        "end_date": (
                            None
                            if finite.is_empty()
                            else finite.get_column("evaluation_date").max()
                        ),
                        **{
                            name: finite.get_column(name).max()
                            if name != "target_start_date"
                            else finite.get_column(name).min()
                            for name in dates
                        },
                    }
                )
        ordered = selected.sort([*groups, "evaluation_date"])
        valid_rolling = (
            ordered.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
            .with_columns(
                pl.col("value")
                .rolling_mean(
                    window_size=rolling_observations,
                    min_samples=rolling_observations,
                )
                .over(*groups)
                .alias("rolling_mean"),
                *(
                    pl.col(name)
                    .cast(pl.Int64)
                    .rolling_max(window_size=rolling_observations, min_samples=1)
                    .over(*groups)
                    .cast(pl.Date)
                    .alias(f"_rolling_{name}")
                    for name in ("target_end_date", "available_date")
                ),
                pl.col("target_start_date")
                .cast(pl.Int64)
                .rolling_min(window_size=rolling_observations, min_samples=1)
                .over(*groups)
                .cast(pl.Date)
                .alias("_rolling_target_start_date"),
            )
            .select(
                "evaluation_date",
                *groups,
                "rolling_mean",
                *(f"_rolling_{name}" for name in dates),
            )
        )
        rolling_frames.append(
            ordered
            .join(
                valid_rolling,
                on=["evaluation_date", *groups],
                how="left",
            )
            .with_columns(
                pl.lit(metric).alias("metric"),
                pl.lit(rolling_observations).alias("rolling_observations"),
                *(
                    pl.coalesce(f"_rolling_{name}", name).alias(name) for name in dates
                ),
                pl.when(
                    pl.col("value").is_null() | ~pl.col("value").is_finite()
                )
                .then(pl.lit("current metric observation is unavailable"))
                .when(pl.col("rolling_mean").is_null())
                .then(pl.lit("insufficient finite observations for rolling window"))
                .when(~pl.col("rolling_mean").is_finite())
                .then(pl.lit("rolling mean is non-finite"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("reason"),
                pl.when(pl.col("rolling_mean").is_finite())
                .then(pl.col("rolling_mean"))
                .otherwise(None)
                .alias("rolling_mean"),
            )
            .with_columns(
                pl.when(pl.col("reason").is_null())
                .then(pl.lit("complete"))
                .otherwise(pl.lit("unavailable"))
                .alias("status")
            )
            .select(
                "evaluation_date",
                "metric",
                "window_kind",
                "window_id",
                "start_session",
                "end_session",
                "rolling_observations",
                "rolling_mean",
                *dates,
                "status",
                "reason",
            )
        )
    stability_schema = {
        "segment": pl.String,
        "metric": pl.String,
        "window_kind": pl.String,
        "window_id": pl.String,
        "start_session": pl.Int64,
        "end_session": pl.Int64,
        "mean": pl.Float64,
        "sample_size": pl.Int64,
        "start_date": pl.Date,
        "end_date": pl.Date,
        **{name: pl.Date for name in dates},
        "status": pl.String,
        "reason": pl.String,
    }
    rolling_schema = {
        "evaluation_date": pl.Date,
        "metric": pl.String,
        "window_kind": pl.String,
        "window_id": pl.String,
        "start_session": pl.Int64,
        "end_session": pl.Int64,
        "rolling_observations": pl.Int32,
        "rolling_mean": pl.Float64,
        **{name: pl.Date for name in dates},
        "status": pl.String,
        "reason": pl.String,
    }
    stability = (
        pl.DataFrame(segment_rows, schema=stability_schema).sort(
            ["metric", "window_id", "segment"]
        )
        if segment_rows
        else pl.DataFrame(schema=stability_schema)
    )
    rolling = (
        pl.concat(rolling_frames, how="vertical_relaxed").sort(
            ["metric", "window_id", "evaluation_date"]
        )
        if rolling_frames
        else pl.DataFrame(schema=rolling_schema)
    )
    return stability, rolling


_SCHEMA = {
    "time": pl.Date,
    "factor": pl.String,
    "exposure": pl.Float64,
    "expected_count": pl.UInt32,
    "observed_count": pl.UInt32,
    "coverage": pl.Float64,
    "expected_weight": pl.Float64,
    "observed_weight": pl.Float64,
    "weight_coverage": pl.Float64,
    "lower": pl.Float64,
    "upper": pl.Float64,
    "constraint_scope": pl.String,
}


def _finish(frame: pl.DataFrame, *, scope: str, bounds: dict) -> pl.DataFrame:
    frame = frame.with_columns(
        pl.col("expected_count", "observed_count").fill_null(0).cast(pl.UInt32),
        pl.col("expected_weight", "observed_weight").fill_null(0.0),
    ).with_columns(
        pl.when(pl.col("observed_count") == pl.col("expected_count"))
        .then(pl.col("exposure").fill_null(0.0))
        .otherwise(None)
        .alias("exposure"),
        pl.when(pl.col("expected_count") == 0)
        .then(1.0)
        .otherwise(pl.col("observed_count") / pl.col("expected_count"))
        .alias("coverage"),
        pl.when(pl.col("expected_weight") == 0)
        .then(1.0)
        .otherwise(pl.col("observed_weight") / pl.col("expected_weight"))
        .alias("weight_coverage"),
        pl.lit(scope).alias("constraint_scope"),
    )
    bound_frame = pl.DataFrame(
        [
            {
                "factor": factor,
                "lower": limits.get("lower"),
                "upper": limits.get("upper"),
            }
            for factor, limits in bounds.items()
        ],
        schema={"factor": pl.String, "lower": pl.Float64, "upper": pl.Float64},
    )
    return (
        frame.join(bound_frame, on="factor", how="left").select(*_SCHEMA).cast(_SCHEMA)
    )


def aggregate_exposures(
    weights: pl.DataFrame,
    coordinates: pl.DataFrame,
    dates: pl.DataFrame,
    *,
    scope: str,
    bounds: dict,
    factors: tuple[str, ...],
) -> pl.DataFrame:
    """Aggregate exact-date coordinates without stock normalization or filling.

    Industry zero is valid only for a known different industry. One missing
    stock coordinate makes that date/factor aggregate null, retaining coverage.
    """
    if dates.is_empty():
        return pl.DataFrame(schema=_SCHEMA)
    if coordinates.select(pl.struct("time", "asset_id").is_duplicated().any()).item():
        raise ValueError("duplicate portfolio risk exposure keys")
    joined = weights.join(coordinates, on=["time", "asset_id"], how="left")
    totals = dates.join(
        weights.group_by("time").agg(
            pl.len().alias("expected_count"),
            pl.col("weight").sum().alias("expected_weight"),
        ),
        on="time",
        how="left",
    )
    outputs = []
    for factor in factors:
        valid = pl.col(factor).is_not_null() & pl.col(factor).is_finite()
        observed = joined.group_by("time").agg(
            (pl.col("weight") * pl.col(factor)).filter(valid).sum().alias("exposure"),
            valid.sum().alias("observed_count"),
            pl.col("weight").filter(valid).sum().alias("observed_weight"),
        )
        outputs.append(
            totals.join(observed, on="time", how="left").with_columns(
                pl.lit(factor).alias("factor")
            )
        )

    known = pl.col("industry").is_not_null() & (
        pl.col("industry").str.strip_chars() != ""
    )
    industry_grid = (
        coordinates.filter(known)
        .select("time", (pl.lit("industry:") + pl.col("industry")).alias("factor"))
        .unique()
    )
    configured = [factor for factor in bounds if factor.startswith("industry:")]
    if configured:
        industry_grid = pl.concat(
            [
                industry_grid,
                dates.join(pl.DataFrame({"factor": configured}), how="cross"),
            ]
        ).unique()
    if not industry_grid.is_empty():
        observed = joined.group_by("time").agg(
            known.sum().alias("observed_count"),
            pl.col("weight").filter(known).sum().alias("observed_weight"),
        )
        industry_values = (
            joined.filter(known)
            .group_by("time", "industry")
            .agg(pl.col("weight").sum().alias("exposure"))
            .with_columns((pl.lit("industry:") + pl.col("industry")).alias("factor"))
            .drop("industry")
        )
        outputs.append(
            industry_grid.join(totals, on="time", how="inner")
            .join(observed, on="time", how="left")
            .join(industry_values, on=["time", "factor"], how="left")
        )
    return _finish(
        pl.concat(outputs, how="diagonal_relaxed"), scope=scope, bounds=bounds
    )


def model_horizon_correlations(components: pl.DataFrame) -> pl.DataFrame:
    """All-row Pearson correlations; a missing component invalidates its pairs."""
    wide = components.select(
        "time", "asset_id", "horizon_sessions", "daily_prediction"
    ).pivot(
        on="horizon_sessions",
        index=["time", "asset_id"],
        values="daily_prediction",
        aggregate_function="first",
    )
    horizon_columns = sorted(
        (column for column in wide.columns if column not in {"time", "asset_id"}),
        key=int,
    )
    correlation_rows: list[dict[str, object]] = []
    if horizon_columns:
        valid = wide.select(
            [
                (pl.col(column).is_not_null() & pl.col(column).is_finite())
                .all()
                .alias(column)
                for column in horizon_columns
            ]
        ).row(0, named=True)
        for first in horizon_columns:
            for second in horizon_columns:
                value = (
                    wide.select(pl.corr(first, second)).item()
                    if valid[first] and valid[second]
                    else None
                )
                correlation_rows.append(
                    {
                        "horizon_sessions": int(first),
                        "other_horizon_sessions": int(second),
                        "correlation": value
                        if value is not None and math.isfinite(value)
                        else None,
                    }
                )
    return pl.DataFrame(
        correlation_rows,
        schema={
            "horizon_sessions": pl.Int64,
            "other_horizon_sessions": pl.Int64,
            "correlation": pl.Float64,
        },
    )


ALPHA_EVALUATION_KERNEL = "alpha.evaluation.v1"


def alpha_evaluation_contract(
    *, quantiles: int, annualization: int, cost_rate: float = 0.0005
) -> dict:
    """Describe authoritative signal mathematics without application policy."""
    from .horizon import DAILY_SESSION_WINDOWS, SIGNAL_PERSISTENCE_HORIZONS

    return {
        "id": ALPHA_EVALUATION_KERNEL,
        "kernel_version": 1,
        "evaluation_frequency": "daily",
        "annualization_sessions": annualization,
        "quantiles": quantiles,
        "factor_return_standardization": "cross_sectional_zscore_ddof0",
        "windows": [
            {
                "window_kind": w.window_kind,
                "window_id": w.window_id,
                "start_session": w.start_session,
                "end_session": w.end_session,
            }
            for w in DAILY_SESSION_WINDOWS
        ],
        "book": {
            "score": "centered_average_rank",
            "gross_exposure": 1.0,
            "net_exposure": 0.0,
        },
        "tail": {
            "long": "q1_equal_weight",
            "short": f"q{quantiles}_equal_weight",
            "gross_exposure": 1.0,
        },
        "signal_persistence_horizons": list(SIGNAL_PERSISTENCE_HORIZONS),
        "daily_rank_path": {
            "rebalance": "caller_selected_sessions",
            "holding": "drift",
            "transaction_cost": {"rate": cost_rate},
            "minimum_fee": "excluded",
            "initial_rebalance_recorded": True,
            "gross_net_ledgers": "independent",
            "negative_lag_semantics": "non_pit_look_ahead",
        },
        "return_input": "caller_aligned_1d_forward_return",
        "return_metadata": ["interval_start", "interval_end", "available_date"],
        "inference": {
            "estimator": "bartlett_newey_west",
            "multiple_testing": "benjamini_hochberg_within_metric_family",
        },
    }


def horizon_tables(
    diagnostics: PredictionHorizonDiagnostics,
) -> dict[str, pl.DataFrame]:
    """Expose partitionable saved diagnostics without price or data discovery."""
    output = {}
    for name in diagnostics.__dataclass_fields__:
        frame = getattr(diagnostics, name)
        if isinstance(frame, pl.DataFrame):
            if "evaluation_date" in frame.columns:
                frame = frame.with_columns(pl.col("evaluation_date").alias("time"))
            output["horizon_" + name] = frame
    return output
