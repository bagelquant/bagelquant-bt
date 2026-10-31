from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bagelquant_bt import (
    InputValidationError,
    build_universe_benchmark_returns,
    compare_portfolio_to_benchmarks,
)
from bagelquant_bt.benchmarks import benchmark_performance


def test_benchmark_performance_preserves_reason_columns():
    performance = benchmark_performance(
        pl.DataFrame({"benchmark": ["cash", "cash"], "return": [0.0, 0.0]}),
        annualization=252,
    )
    assert performance.item(0, "sharpe") is None
    assert performance.item(0, "sharpe_reason") == "sample variance is zero"
    assert performance.schema["annualized_return_reason"] == pl.String
    assert performance.item(0, "annualized_return_reason") is None


def test_benchmark_comparison_retains_finite_path_when_annualization_overflows():
    day = date(2024, 1, 2)
    paths, summary = compare_portfolio_to_benchmarks(
        pl.DataFrame({"time": [day], "return": [0.0]}),
        pl.DataFrame({"time": [day], "benchmark": ["market"], "return": [-0.95]}),
    )
    assert paths["relative_wealth"][0] == pytest.approx(20)
    assert summary["annualized_excess_return"][0] is None
    assert (
        summary["annualized_excess_return_reason"][0]
        == "annualized wealth is non-finite"
    )
    assert summary["max_relative_drawdown"][0] == 0


def test_portfolio_benchmark_comparison_uses_exact_aligned_returns() -> None:
    portfolio = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "return": [0.10, -0.05, 0.02],
        }
    )
    benchmarks = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-04"],
            "benchmark": ["market", "market", "market"],
            "return": [0.05, 0.0, 0.01],
        }
    )

    paths, summary = compare_portfolio_to_benchmarks(
        portfolio, benchmarks, annualization=2
    )

    expected_relative = (1.10 * 0.95) / (1.05 * 1.0)
    assert paths.get_column("time").to_list() == [
        date(2024, 1, 1),
        date(2024, 1, 2),
    ]
    assert paths.get_column("relative_wealth")[-1] == pytest.approx(expected_relative)
    assert summary.item(0, "annualized_excess_return") == pytest.approx(
        expected_relative - 1.0
    )
    assert summary.item(0, "tracking_error") == pytest.approx(0.1)
    assert summary.item(0, "information_ratio") == pytest.approx(0.0)
    assert summary.item(0, "daily_win_rate") == pytest.approx(0.5)
    assert summary.item(0, "max_relative_drawdown") == pytest.approx(
        expected_relative / (1.10 / 1.05) - 1.0
    )


def test_portfolio_benchmark_comparison_handles_constant_and_empty_alignment() -> None:
    portfolio = pl.DataFrame(
        {"time": ["2024-01-01", "2024-01-02"], "return": [0.01, 0.01]}
    )
    constant = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "benchmark": ["same", "same"],
            "return": [0.01, 0.01],
        }
    )
    _, summary = compare_portfolio_to_benchmarks(portfolio, constant)
    assert summary.item(0, "tracking_error") == 0.0
    assert summary.item(0, "information_ratio") is None

    unmatched = constant.with_columns(pl.col("time").str.replace("2024", "2025"))
    paths, summary = compare_portfolio_to_benchmarks(portfolio, unmatched)
    assert paths.is_empty()
    assert summary.is_empty()


def test_universe_benchmark_renormalizes_available_equal_and_size_samples() -> None:
    returns = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "forward_return": [0.10, -0.10],
        }
    )
    universe = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 3,
            "asset_id": ["a", "b", "missing"],
        }
    )
    sizes = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "size": [3.0, 1.0],
        }
    )

    equal, equal_coverage = build_universe_benchmark_returns(returns, universe=universe)
    cap, cap_coverage = build_universe_benchmark_returns(
        returns, universe=universe, sizes=sizes, name="cap"
    )

    assert equal["return"].to_list() == pytest.approx([0.0])
    assert cap["return"].to_list() == pytest.approx([0.05])
    assert equal_coverage.select(
        "expected_count", "observed_count", "coverage_ratio"
    ).to_dicts() == [
        {
            "expected_count": 3,
            "observed_count": 2,
            "coverage_ratio": pytest.approx(2 / 3),
        }
    ]
    assert (
        cap_coverage.select(
            "expected_count", "observed_count", "coverage_ratio"
        ).to_dicts()
        == equal_coverage.select(
            "expected_count", "observed_count", "coverage_ratio"
        ).to_dicts()
    )


@pytest.mark.parametrize("returns", [[-0.1, 0.0], [-1.0, 0.0], [-0.1, 0.2, -0.05]])
def test_relative_drawdown_retains_initial_wealth_and_bankruptcy(returns):
    from datetime import timedelta

    days = [date(2024, 1, 2) + timedelta(days=index) for index in range(len(returns))]
    paths, summary = compare_portfolio_to_benchmarks(
        pl.DataFrame({"time": days, "return": returns}),
        pl.DataFrame({"time": days, "benchmark": ["flat"] * len(days),
                      "return": [0.0] * len(days)}),
    )
    wealth, peak, expected = 1.0, 1.0, []
    for value in returns:
        wealth *= 1 + value
        peak = max(peak, wealth)
        expected.append(wealth / peak - 1)
    assert paths["relative_drawdown"].to_list() == pytest.approx(expected)
    assert summary.item(0, "max_relative_drawdown") == pytest.approx(min(expected))


def test_saved_portfolio_benchmark_window_rebases_its_initial_relative_peak():
    from bagelquant_bt import BENCHMARK_COMPARISON_POLICY, aggregate_saved_window

    days = [date(2024, 1, 2), date(2024, 1, 3)]
    metrics, tables = aggregate_saved_window(
        "portfolio", "benchmark",
        {"benchmark_comparison": pl.DataFrame({"time": days,
            "portfolio_return": [-0.1, 0.0]}),
         "benchmark_returns": pl.DataFrame({"time": days,
            "benchmark": ["flat", "flat"], "return": [0.0, 0.0]})},
        {"annualization": 252}, start=days[0], end=days[-1],
    )
    assert tables["benchmark_comparison"]["relative_drawdown"].to_list() == (
        pytest.approx([-0.1, -0.1])
    )
    assert tables["benchmark_statistics"].item(0, "max_relative_drawdown") == (
        pytest.approx(-0.1)
    )
    assert metrics["benchmark_contract"] == BENCHMARK_COMPARISON_POLICY


@pytest.mark.parametrize("explicit_universe", [False, True])
@pytest.mark.parametrize("size_weighted", [False, True])
def test_universe_benchmarks_reject_duplicate_return_coordinates(
    explicit_universe, size_weighted,
):
    day = date(2024, 1, 2)
    returns = pl.DataFrame({"time": [day] * 3, "asset_id": ["a", "a", "b"],
                            "forward_return": [0.1, 0.2, 0.0]})
    universe = pl.DataFrame({"time": [day] * 2, "asset_id": ["a", "b"]})
    sizes = universe.with_columns(pl.lit(1.0).alias("size"))
    with pytest.raises(InputValidationError, match="forward_returns must be unique"):
        build_universe_benchmark_returns(returns,
            universe=universe if explicit_universe else None,
            sizes=sizes if size_weighted else None)


@pytest.mark.parametrize("explicit_universe", [False, True])
@pytest.mark.parametrize("size_weighted", [False, True])
@pytest.mark.parametrize(("column", "value"), [
    ("time", None), ("time", "not-a-date"), ("asset_id", None),
])
def test_universe_benchmarks_reject_missing_or_invalid_return_keys(
    explicit_universe, size_weighted, column, value,
):
    values = {"time": ["2024-01-02"] * 2, "asset_id": ["a", "b"],
              "forward_return": [0.1, 0.0]}
    values[column][0] = value
    universe = pl.DataFrame({"time": ["2024-01-02"] * 2, "asset_id": ["a", "b"]})
    sizes = universe.with_columns(pl.lit(1.0).alias("size"))
    with pytest.raises(InputValidationError, match="forward_returns require nonnull"):
        build_universe_benchmark_returns(pl.DataFrame(values),
            universe=universe if explicit_universe else None,
            sizes=sizes if size_weighted else None)


@pytest.mark.parametrize("explicit_universe", [False, True])
@pytest.mark.parametrize("size_weighted", [False, True])
def test_universe_benchmark_key_checks_preserve_missing_return_semantics(
    explicit_universe, size_weighted,
):
    returns = pl.DataFrame({"time": ["2024-01-02"] * 2,
        "asset_id": [1, 2], "forward_return": [0.1, None]})
    universe = pl.DataFrame({"time": ["2024-01-02"] * 2, "asset_id": [1, 2]})
    sizes = universe.with_columns(pl.lit(1.0).alias("size"))
    benchmark, coverage = build_universe_benchmark_returns(returns,
        universe=universe if explicit_universe else None,
        sizes=sizes if size_weighted else None)
    assert benchmark["return"].to_list() == pytest.approx([0.1])
    assert coverage["coverage_ratio"].to_list() == [0.5]
