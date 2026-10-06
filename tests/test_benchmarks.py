from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bagelquant_bt import (
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
