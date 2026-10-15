from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt import horizon as horizon_module
from bagelquant_bt import signal_rank_persistence, summarize_signal_persistence


def _eager_reference(factor, calendar, horizons):
    """Retain the original joined-universe formula as an independent control."""
    normalized = (
        factor.select("evaluation_date", "execution_date", "asset_id", "factor")
        .drop_nulls()
        .filter(pl.col("factor").is_finite())
        .sort("evaluation_date", "asset_id")
    )
    sessions = calendar.filter(pl.col("is_open"))["time"].sort().to_list()
    indexed = normalized.join(
        pl.DataFrame({"evaluation_date": sessions, "ordinal": range(len(sessions))}),
        on="evaluation_date",
    )
    future = indexed.select(
        pl.col("evaluation_date").alias("target_date"),
        "asset_id",
        pl.col("factor").alias("future_factor"),
        pl.col("ordinal").alias("target_ordinal"),
    )
    frames = []
    for lag in horizons:
        paired = (
            indexed.with_columns((pl.col("ordinal") + lag).alias("target_ordinal"))
            .join(future, on=["target_ordinal", "asset_id"])
            .with_columns(
                pl.lit(lag, dtype=pl.Int64).alias("horizon_sessions"),
                pl.col("factor").rank("average").over("evaluation_date").alias("x"),
                pl.col("future_factor")
                .rank("average")
                .over("evaluation_date")
                .alias("y"),
            )
        )
        frames.append(
            paired.group_by("evaluation_date", "target_date", "horizon_sessions").agg(
                pl.when(
                    (pl.len() < 2)
                    | (pl.col("x").n_unique() < 2)
                    | (pl.col("y").n_unique() < 2)
                )
                .then(None)
                .otherwise(pl.corr("x", "y"))
                .alias("rank_autocorrelation"),
                pl.len().alias("sample_size"),
            )
        )
    return pl.concat(frames).sort("evaluation_date", "horizon_sessions")


@pytest.mark.parametrize("row_budget", [1, 6, 250_000])
@pytest.mark.parametrize("continued", [False, True])
def test_persistence_matches_original_joined_universe_across_chunks(
    row_budget, continued, monkeypatch
):
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(7)]
    crosses = [
        {"a": 0.0, "b": 10.0, "c": 20.0, "x": 5.0, "nan": float("nan")},
        {"a": 3.0, "b": 1.0, "c": 2.0, "y": 10.0, "null": None},
        {"a": 10.0, "b": 0.0, "c": 1.0},
        {"a": 1.0, "b": 1.0, "c": 4.0, "inf": float("inf")},
        {"a": 2.0},
        {"a": 2.0, "b": 2.0, "c": 2.0},
        {},
    ]
    factor = pl.DataFrame(
        [
            (day, day, asset, value)
            for day, cross in zip(days, crosses, strict=True)
            for asset, value in cross.items()
        ],
        schema={
            "evaluation_date": pl.Date,
            "execution_date": pl.Date,
            "asset_id": pl.String,
            "factor": pl.Float64,
        },
        orient="row",
    ).reverse()
    calendar = pl.DataFrame({"time": days, "is_open": [True, True, False] + [True] * 4})
    horizons = (1, 2, 5)
    expected = _eager_reference(factor, calendar, horizons)
    target_start = days[3] if continued else None
    if target_start is not None:
        expected = expected.filter(pl.col("target_date") > target_start)
    monkeypatch.setattr(horizon_module, "_PERSISTENCE_PAIR_ROW_BUDGET", row_budget)
    progress = []
    series, summary = signal_rank_persistence(
        factor,
        calendar=calendar,
        horizons=horizons,
        target_start=target_start,
        progress=lambda *args: progress.append(args),
    )
    assert_frame_equal(series, expected, check_exact=True)
    assert_frame_equal(
        summary, summarize_signal_persistence(expected, horizons), check_exact=True
    )
    if not continued:
        # Ranks must be recalculated after excluding x/y, rather than before joining.
        assert (
            series.filter(
                (pl.col("evaluation_date") == days[0])
                & (pl.col("horizon_sessions") == 1)
            )["rank_autocorrelation"].item()
            == -0.5
        )
    assert progress == [("signal_autocorrelation", index, 3) for index in (1, 2, 3)]


def test_public_persistence_still_rejects_duplicate_finite_coordinates():
    factor = pl.DataFrame(
        {
            "evaluation_date": [date(2024, 1, 1)] * 2,
            "execution_date": [date(2024, 1, 1)] * 2,
            "asset_id": ["a", "a"],
            "factor": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        signal_rank_persistence(
            factor, calendar=pl.DataFrame({"time": factor["evaluation_date"]})
        )
