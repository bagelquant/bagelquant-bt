from datetime import date, timedelta
from statistics import correlation

import polars as pl
import pytest

from bagelquant_bt import pair_correlation

START = date(2020, 1, 1)
END = date(2020, 2, 1)


def series(values, offset=0):
    return pl.DataFrame(
        {
            "time": [START + timedelta(days=i + offset) for i in range(len(values))],
            "value": values,
        },
        schema={"time": pl.Date, "value": pl.Float64},
    )


def panel(values, day=START):
    return pl.DataFrame(
        {
            "time": [day] * len(values),
            "asset_id": [str(i) for i in range(len(values))],
            "value": values,
        },
        schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64},
    )


def test_returns_append_repair_delete_noop_equal_independent_reference():
    x, y = [float(i) for i in range(13)], [float(i % 4) for i in range(13)]
    dates = [START + timedelta(days=i) for i in range(13)]
    first = pair_correlation(
        series(x[:12]), series(y[:12]), changed_dates=dates[:12], start=START, end=END
    )
    appended = pair_correlation(
        series(x[12:], 12),
        series(y[12:], 12),
        changed_dates=dates[12:],
        previous=first["state"],
        start=START,
        end=END,
    )
    assert appended["correlation"] == pytest.approx(correlation(x, y))
    assert appended["processed_dates"] == 1
    assert first["processed_dates"] == 12
    y[4] = 15.0
    repaired = pair_correlation(
        series([x[4]], 4),
        series([y[4]], 4),
        changed_dates=[dates[4]],
        previous=appended["state"],
        start=START,
        end=END,
    )
    assert repaired["correlation"] == pytest.approx(correlation(x, y))
    assert repaired["processed_dates"] == 1
    empty = series([])
    deleted = pair_correlation(
        empty,
        empty,
        changed_dates=[dates[4]],
        previous=repaired["state"],
        start=START,
        end=END,
    )
    assert deleted["correlation"] == pytest.approx(
        correlation(x[:4] + x[5:], y[:4] + y[5:])
    )
    noop = pair_correlation(
        empty, empty, changed_dates=[], previous=deleted["state"], start=START, end=END
    )
    assert noop["correlation"] == deleted["correlation"]
    assert noop["processed_dates"] == 0


def test_daily_cross_sections_equal_weight_mean_and_tied_rank_reference():
    next_day = START + timedelta(days=1)
    result = pair_correlation(
        pl.concat([panel([1, 2, 3]), panel([1, 2, 3], next_day)]),
        pl.concat([panel([1, 2, 3]), panel([3, 2, 1], next_day)]),
        kind="cross_section",
        changed_dates=[START, next_day],
        start=START,
        end=END,
    )
    assert result["correlation"] == pytest.approx(0)
    assert result["sample_count"] == 2
    tied = pair_correlation(
        panel([1, 1, 2, 3]),
        panel([3, 2, 2, 1]),
        kind="cross_section",
        method="spearman",
        changed_dates=[START],
        start=START,
        end=END,
    )
    assert tied["correlation"] == pytest.approx(-5 / 6)


def test_complete_holdings_union_zero_fills_only_absent_assets():
    left = panel([0.5, 0.5])
    right = panel([1.0]).with_columns(pl.lit("2").alias("asset_id"))
    result = pair_correlation(
        left,
        right,
        kind="cross_section",
        zero_missing=True,
        changed_dates=[START],
        start=START,
        end=END,
    )
    assert result["correlation"] == pytest.approx(-1)
    null = pl.concat([left, panel([None]).with_columns(pl.lit("2").alias("asset_id"))])
    missing = pair_correlation(
        null,
        right,
        kind="cross_section",
        zero_missing=True,
        changed_dates=[START],
        start=START,
        end=END,
    )
    assert missing["correlation"] is None  # Present missing mark is never unheld.
    idle = panel([0.0, 0.0]).with_columns(pl.Series("asset_id", ["3", "4"]))
    retained_exits = pair_correlation(
        pl.concat([left, idle]),
        pl.concat([right, idle]),
        kind="cross_section",
        zero_missing=True,
        changed_dates=[START],
        start=START,
        end=END,
    )
    assert retained_exits["correlation"] == pytest.approx(-1)
    assert retained_exits["state"]["days"][0]["sample_count"] == 3


@pytest.mark.parametrize("values", [[1.0] * 12, [None] * 12, [float("nan")] * 12, []])
def test_undefined_returns_preserve_null_and_reason(values):
    dates = [START + timedelta(days=i) for i in range(len(values))]
    result = pair_correlation(
        series(values), series(values), changed_dates=dates, start=START, end=END
    )
    assert result["correlation"] is None
    assert result["reason"] in {"constant_series", "insufficient_samples"}


def test_duplicate_coordinates_wrong_types_and_unadmitted_dates_rejected():
    with pytest.raises(ValueError, match="unique"):
        pair_correlation(
            pl.concat([panel([1, 2, 3])] * 2),
            panel([1, 2, 3]),
            kind="cross_section",
            changed_dates=[START],
            start=START,
            end=END,
        )
    with pytest.raises(ValueError, match="admitted"):
        pair_correlation(
            series([1]), series([1]), changed_dates=[], start=START, end=END
        )
    with pytest.raises(ValueError, match="nonnull Date"):
        pair_correlation(
            series([1]).with_columns(pl.col("time").cast(pl.String)),
            series([1]),
            changed_dates=[START],
            start=START,
            end=END,
        )


def test_sparse_finite_asset_join_precedes_spearman_ranking_and_empty_repair():
    left_frame, right_frame = (
        panel([1, 2, None, 3, float("inf")]),
        panel([3, 2, 10, 1, 1]),
    )
    result = pair_correlation(
        left_frame,
        right_frame,
        kind="cross_section",
        method="spearman",
        changed_dates=[START],
        start=START,
        end=END,
    )
    assert result["correlation"] == pytest.approx(-1)
    repaired = pair_correlation(
        left_frame.head(0),
        right_frame.head(0),
        kind="cross_section",
        method="spearman",
        changed_dates=[START],
        previous=result["state"],
        start=START,
        end=END,
    )
    assert repaired["correlation"] is None
    assert repaired["processed_dates"] == 1
