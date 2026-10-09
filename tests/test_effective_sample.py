from datetime import date, timedelta

import polars as pl
import pytest

from bagelquant_bt import evaluation_sample
from bagelquant_bt.periods import aggregate_saved_window
from bagelquant_bt.statistics import return_metrics


def test_leading_blanks_do_not_change_economic_statistics_but_later_gaps_remain():
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(6)]
    returns = pl.DataFrame(
        {"time": days, "net_return": [0.0, 0.0, 0.0, None, -0.1, 0.2]}
    )
    coverage = pl.DataFrame(
        {"time": days, "signal_count": [0, 0, 2, 0, 2, 2], "universe_count": [2] * 6}
    )
    frames = {"daily_signal_coverage": coverage, "daily_book_returns": returns}
    metadata = evaluation_sample(
        frames, start=days[0], end=days[-1], minimum_history_years=5
    )
    assert metadata["effective_start"] == str(days[2])
    assert metadata["excluded_leading_observations"] == 2
    assert metadata["valid_observations"] == 3
    assert metadata["missing_ratio"] == 0.25
    assert metadata["warnings"] == ["effective_history_below_minimum"]
    metrics, _ = aggregate_saved_window(
        "alpha",
        "summary",
        frames,
        {"annualization": 240},
        start=days[0],
        end=days[-1],
        items=("signal_coverage",),
    )
    trimmed = {
        name: frame.filter(pl.col("time") >= days[2]) for name, frame in frames.items()
    }
    reference, _ = aggregate_saved_window(
        "alpha",
        "summary",
        trimmed,
        {"annualization": 240},
        start=days[2],
        end=days[-1],
        items=("signal_coverage",),
    )
    assert metrics["book"] == reference["book"]
    assert metrics["book"]["net_sample_size"] == 3


def test_calendar_anniversary_and_observation_count_without_signal_coverage():
    frame = pl.DataFrame(
        {"time": [date(2019, 1, 1), date(2024, 1, 1)], "net_return": [0.0, 0.01]}
    )
    metadata = evaluation_sample(
        {"daily_book_returns": frame},
        start=date(2019, 1, 1),
        end=date(2024, 1, 1),
        minimum_history_years=5,
    )
    assert metadata["warnings"] == []
    assert metadata["valid_observations"] == 2
    assert metadata["calendar_basis"] == "saved_observations"
    no_signal = evaluation_sample(
        {
            "daily_signal_coverage": pl.DataFrame(
                {"time": [date(2020, 1, 1)], "signal_count": [0]}
            )
        },
        start=date(2020, 1, 1),
        end=date(2020, 1, 2),
    )
    assert no_signal["effective_start"] is None
    metrics, _ = aggregate_saved_window(
        "alpha",
        "summary",
        {
            "daily_signal_coverage": pl.DataFrame(
                {"time": [date(2020, 1, 1)], "signal_count": [0]}
            ),
            "daily_book_returns": pl.DataFrame(
                {"time": [date(2020, 1, 1)], "net_return": [0.0]}
            ),
        },
        {"annualization": 240},
        start=date(2020, 1, 1),
        end=date(2020, 1, 2),
    )
    assert metrics["book"]["net_sample_size"] == 0
    assert metrics["book"]["net_annualized_return"] is None


def test_expected_shortfall_uses_worst_five_percent_including_zero():
    assert return_metrics([-0.2, -0.1] + [0.0] * 19, 240)[
        "expected_shortfall_95"
    ] == pytest.approx(0.15)
    assert return_metrics([-0.2] + [0.0] * 19, 240)[
        "expected_shortfall_95"
    ] == pytest.approx(0.2)
    assert return_metrics([0.0] * 20, 240)["expected_shortfall_95"] == 0.0
    assert return_metrics([0.0] * 19, 240)["expected_shortfall_95"] is None


def test_unmatured_economic_observations_cannot_start_the_sample():
    frame = pl.DataFrame(
        {
            "time": [date(2024, 1, 1)],
            "net_return": [0.0],
            "interval_end": [date(2024, 1, 3)],
            "available_date": [date(2024, 1, 3)],
        }
    )
    result = evaluation_sample(
        {"daily_book_returns": frame}, start=date(2024, 1, 1), end=date(2024, 1, 2)
    )
    assert result["effective_start"] is None
    assert result["first_return_date"] is None
    assert result["warnings"] == ["no_available_observations"]


def test_ic_period_summary_discloses_annual_and_recent_sample():
    frame = pl.DataFrame(
        {
            "evaluation_date": [date(2019, 1, 1), date(2020, 1, 1), date(2024, 1, 1)],
            "window_kind": ["cumulative"] * 3,
            "window_id": ["cumulative_1d"] * 3,
            "start_session": [1] * 3,
            "end_session": [1] * 3,
            "pearson_ic": [-0.5, 0.1, 0.3],
            "spearman_ic": [-0.5, 0.1, 0.3],
        }
    )
    _, tables = aggregate_saved_window(
        "alpha",
        "ic_horizon_profile",
        {"horizon_ic": frame},
        {"annualization": 240},
        start=date(2000, 1, 1),
        end=date(2024, 12, 31),
    )
    assert tables["annual_ic"]["year"].unique().sort().to_list() == [2019, 2020, 2024]
    recent = (
        tables["recent_ic"].filter(pl.col("method") == "spearman").row(0, named=True)
    )
    assert recent["start"] == date(2020, 1, 1)
    assert recent["sample_size"] == 2
    assert recent["mean"] == pytest.approx(0.2)
