from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bagelquant_bt import (
    BacktestConfig,
    build_universe_benchmark_returns,
    compare_portfolio_to_benchmarks,
    summary_report,
)
from bagelquant_bt.factor import run_factor_evaluation


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

    equal, equal_coverage = build_universe_benchmark_returns(
        returns, universe=universe
    )
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
    assert cap_coverage.select(
        "expected_count", "observed_count", "coverage_ratio"
    ).to_dicts() == equal_coverage.select(
        "expected_count", "observed_count", "coverage_ratio"
    ).to_dicts()


def test_factor_result_contains_default_external_and_both_excess_paths() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
            "asset_id": ["a"] * 3 + ["b"] * 3,
            "price": [100.0, 110.0, 121.0, 100.0, 100.0, 100.0],
        }
    )
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "factor": [2.0, 1.0],
        }
    )
    external = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "benchmark": ["external", "external"],
            "return": [0.05, 0.05],
        }
    )
    external_coverage = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "benchmark": ["external", "external"],
            "expected_count": [3, 3],
            "observed_count": [3, 2],
            "coverage_ratio": [1.0, 2 / 3],
        }
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(
            initial_capital=1_000_000, quantiles=2, top_n=1
        ),
        benchmark_returns=external,
        benchmark_coverage=external_coverage,
    )

    assert set(result.benchmark_returns["benchmark"]) == {
        "universe_equal_weight",
        "external",
    }
    assert set(result.benchmark_performance["benchmark"]) == {
        "universe_equal_weight",
        "external",
    }
    external_net = result.excess_returns.filter(
        (pl.col("benchmark") == "external") & (pl.col("portfolio") == "net")
    )
    expected_first = (
        result.top_n_backtest.returns["net_return"][0] - external["return"][0]
    )
    assert external_net["daily_excess_return"][0] == pytest.approx(expected_first)
    assert external_net.columns == [
        "time",
        "benchmark",
        "portfolio",
        "portfolio_return",
        "benchmark_return",
        "daily_excess_return",
        "compounded_excess_return",
        "relative_wealth_excess_return",
    ]


def test_external_returns_infer_complete_series_coverage() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
            "asset_id": ["a"] * 3 + ["b"] * 3,
            "price": [100.0, 110.0, 121.0, 100.0, 100.0, 100.0],
        }
    )
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "factor": [2.0, 1.0],
        }
    )
    external = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "benchmark": ["external", "external"],
            "return": [0.01, -0.01],
        }
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(
            initial_capital=10_000,
            quantiles=2,
            top_n=1,
        ),
        benchmark_returns=external,
    )

    coverage = result.benchmark_coverage.filter(
        pl.col("benchmark") == "external"
    )
    assert coverage.get_column("coverage_ratio").to_list() == [1.0, 1.0]


def test_report_warns_when_external_benchmark_coverage_is_incomplete() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
            "asset_id": ["a"] * 3 + ["b"] * 3,
            "price": [100.0, 110.0, 121.0, 100.0, 100.0, 100.0],
        }
    )
    factor = prices.filter(pl.col("time") == "2024-01-01").select(
        "time",
        "asset_id",
        pl.when(pl.col("asset_id") == "a")
        .then(2.0)
        .otherwise(1.0)
        .alias("factor"),
    )
    external = pl.DataFrame(
        {
            "time": ["2024-01-01"],
            "benchmark": ["external"],
            "return": [0.01],
        }
    )
    coverage = pl.DataFrame(
        {
            "time": ["2024-01-01"],
            "benchmark": ["external"],
            "expected_count": [3],
            "observed_count": [2],
            "coverage_ratio": [2 / 3],
        }
    )
    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(
            initial_capital=10_000,
            quantiles=2,
            top_n=1,
        ),
        benchmark_returns=external,
        benchmark_coverage=coverage,
    )

    html = summary_report(result)

    assert "Coverage warning:" in html
    assert "index returns were not forward-filled" in html
