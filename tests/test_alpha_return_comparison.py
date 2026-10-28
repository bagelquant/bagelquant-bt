"""Alpha Return compares saved lag paths on their common economic sample."""

from datetime import date, timedelta
from math import sqrt
from statistics import mean, stdev

import polars as pl
import pytest

from bagelquant_bt import aggregate_saved_window, alpha_return_comparison
from bagelquant_bt.exceptions import InputValidationError


def saved_paths():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(8)]
    daily = pl.DataFrame(
        {
            "time": days[:6],
            "interval_end": days[2:],
            "available_date": days[2:],
            "gross_return": [-0.04, -0.02, 0.03, 0.01, -0.01, 0.02],
            "net_return": [-0.041, -0.021, 0.029, 0.009, -0.011, 0.019],
        }
    )
    lagged = pl.concat(
        [
            daily.tail(4).with_columns(
                pl.lit(lag).alias("lag"), pl.lit(path).alias("path_kind")
            )
            for path in ("book", "tail")
            for lag in (0, 1)
        ]
    )
    return days, {
        "daily_book_returns": daily,
        "daily_tail_returns": daily,
        "daily_alpha_return_lag_returns": lagged,
    }


@pytest.mark.parametrize("annualization", [12, 240, 252])
def test_table_statistics_use_common_lag_zero_not_full_daily_path(annualization):
    days, frames = saved_paths()
    metrics, tables = aggregate_saved_window(
        "alpha",
        "alpha_return",
        frames,
        {"annualization": annualization},
        start=days[0],
        end=days[-1],
    )
    for path in ("book", "tail"):
        for kind in ("gross", "net"):
            values = frames[f"daily_{path}_returns"][f"{kind}_return"].tail(4)
            expected = mean(values) / stdev(values) * sqrt(annualization)
            assert metrics[path][f"{kind}_sharpe"] == pytest.approx(expected)
            assert metrics[path][f"{kind}_sample_size"] == 4
        assert tables[f"{path}_daily_returns"]["time"].to_list() == days[2:6]
    assert metrics["comparison_sample"] == {
        "basis": "common_lag_dates",
        "start": str(days[2]),
        "end": str(days[5]),
        "observations": 4,
        "annualization": annualization,
    }


def test_common_sample_respects_maturity_and_preserves_zero_return():
    days, frames = saved_paths()
    frames["daily_alpha_return_lag_returns"] = frames[
        "daily_alpha_return_lag_returns"
    ].with_columns(
        pl.when(pl.col("time") == days[2])
        .then(0.0)
        .otherwise(pl.col("net_return"))
        .alias("net_return")
    )
    metrics, tables = aggregate_saved_window(
        "prediction",
        "alpha_return",
        frames,
        {"annualization": 240},
        start=days[2],
        end=days[5],
    )
    assert tables["book_daily_returns"]["time"].to_list() == days[2:4]
    assert metrics["book"]["net_sample_size"] == 2
    assert metrics["comparison_sample"]["end"] == str(days[3])
    assert tables["book_daily_returns"]["net_return"][0] == 0.0


@pytest.mark.parametrize("count", [0, 1])
def test_insufficient_comparison_does_not_fall_back_to_full_returns(count):
    days, frames = saved_paths()
    frames["daily_alpha_return_lag_returns"] = frames[
        "daily_alpha_return_lag_returns"
    ].filter(pl.col("time").is_in(days[2 : 2 + count]))
    metrics, _ = aggregate_saved_window(
        "alpha",
        "alpha_return",
        frames,
        {"annualization": 240},
        start=days[0],
        end=days[-1],
    )
    assert metrics["book"]["net_sharpe"] is None
    assert metrics["book"]["net_sample_size"] == count


def test_saved_path_append_and_repair_summaries_match_independent_reference():
    days, original = saved_paths()
    for change in ("append", "repair"):
        frames = dict(original)
        paths = original["daily_alpha_return_lag_returns"]
        if change == "append":
            paths = pl.concat(
                [
                    paths,
                    paths.filter(pl.col("time") == days[5]).with_columns(
                        pl.lit(days[6]).alias("time"),
                        pl.lit(days[7]).alias("interval_end"),
                        pl.lit(days[7]).alias("available_date"),
                    ),
                ]
            )
        else:
            paths = paths.with_columns(
                pl.when(pl.col("time") == days[3])
                .then(-0.03)
                .otherwise(pl.col("net_return"))
                .alias("net_return")
            )
        frames["daily_alpha_return_lag_returns"] = paths
        metrics, _ = aggregate_saved_window(
            "alpha",
            "alpha_return",
            frames,
            {"annualization": 240},
            start=days[0],
            end=days[-1],
        )
        values = paths.filter((pl.col("lag") == 0) & (pl.col("path_kind") == "book"))[
            "net_return"
        ]
        assert metrics["book"]["net_sharpe"] == pytest.approx(
            mean(values) / stdev(values) * sqrt(240)
        )


def test_public_comparison_preserves_sparse_dates_nulls_and_missing_lag_zero():
    days, frames = saved_paths()
    paths = frames["daily_alpha_return_lag_returns"].filter(pl.col("time") != days[3])
    paths = paths.with_columns(
        pl.when(pl.col("time") == days[2])
        .then(None)
        .otherwise(pl.col("net_return"))
        .alias("net_return")
    )
    metrics, tables = alpha_return_comparison(paths, annualization=240)
    assert metrics["comparison_sample"]["observations"] == 3
    assert metrics["book"]["net_sample_size"] == 2
    assert tables["book_daily_returns"]["time"].to_list() == [days[2], days[4], days[5]]
    missing, _ = alpha_return_comparison(
        paths.filter(pl.col("lag") != 0), annualization=240
    )
    assert missing["book"]["net_sample_size"] == 0
    assert missing["book"]["net_sharpe"] is None


def test_public_comparison_rejects_invalid_and_duplicate_coordinates():
    _, frames = saved_paths()
    paths = frames["daily_alpha_return_lag_returns"]
    for invalid in (
        pl.concat([paths, paths.head(1)]),
        paths.with_columns(pl.lit("1").alias("lag")),
        paths.with_columns(pl.lit("foreign").alias("path_kind")),
    ):
        with pytest.raises(InputValidationError, match="coordinates"):
            alpha_return_comparison(invalid, annualization=240)
