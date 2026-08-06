from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bagelquant_bt import (
    RESULT_SECTION_VERSION,
    BacktestConfig,
    PortfolioPathIdentity,
    ResultSectionSpec,
    ResultWindow,
    compute_result_section,
    materialize_portfolio_path,
    resume_portfolio_path,
)


def test_result_section_version_is_public() -> None:
    assert RESULT_SECTION_VERSION == 4


def _prices() -> pl.DataFrame:
    days = [f"2024-01-0{day}" for day in range(1, 6)]
    return pl.DataFrame(
        {
            "time": days * 2,
            "asset_id": ["a"] * 5 + ["b"] * 5,
            "price": [10.0, 11.0, 12.0, 13.0, 14.0, 10.0, 10.0, 11.0, 12.0, 13.0],
        }
    )


def _weights() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-04",
                "2024-01-04",
            ],
            "asset_id": ["a", "b"] * 3,
            "weight": [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        }
    )


def _availability() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": ["2024-01-02"],
            "asset_id": ["b"],
            "can_buy": [False],
            "can_sell": [True],
            "reason": ["limit_up"],
        }
    )


def _identity() -> PortfolioPathIdentity:
    return PortfolioPathIdentity(
        alpha_revision="alpha-v1",
        universe="u",
        policy_combo="combo-v2",
    )


def test_resumed_path_matches_one_shot_path_across_pending_trade() -> None:
    prices = _prices()
    weights = _weights()
    availability = _availability()
    config = BacktestConfig(initial_capital=100_000, annualization=4)

    whole = materialize_portfolio_path(
        weights,
        prices,
        identity=_identity(),
        config=config,
        execution_availability=availability,
    )
    first = materialize_portfolio_path(
        weights.filter(pl.col("time") <= "2024-01-02"),
        prices.filter(pl.col("time") <= "2024-01-03"),
        identity=_identity(),
        config=config,
        execution_availability=availability,
    )
    second = resume_portfolio_path(
        weights.filter(pl.col("time") >= "2024-01-04"),
        prices.filter(pl.col("time") >= "2024-01-03"),
        identity=_identity(),
        checkpoint=first.checkpoint,
        config=config,
    )

    resumed_returns = pl.concat([first.returns, second.returns]).sort("time")
    resumed_turnover = pl.concat([first.turnover, second.turnover]).sort("time")
    resumed_costs = pl.concat([first.costs, second.costs]).sort("time")
    assert resumed_returns.equals(whole.returns)
    assert resumed_turnover.equals(whole.turnover)
    assert resumed_costs.equals(whole.costs)
    assert second.checkpoint.net_value == whole.checkpoint.net_value


def test_resume_replaces_sparse_target_at_checkpoint_time() -> None:
    prices = _prices()
    weights = _weights().filter(pl.col("weight") > 0.0)
    config = BacktestConfig(initial_capital=100_000, annualization=4)
    whole = materialize_portfolio_path(
        weights,
        prices,
        identity=_identity(),
        config=config,
    )
    first = materialize_portfolio_path(
        weights.filter(pl.col("time") < "2024-01-02"),
        prices.filter(pl.col("time") <= "2024-01-02"),
        identity=_identity(),
        config=config,
    )
    second = resume_portfolio_path(
        weights.filter(pl.col("time") >= "2024-01-02"),
        prices.filter(pl.col("time") >= "2024-01-02"),
        identity=_identity(),
        checkpoint=first.checkpoint,
        config=config,
    )

    assert pl.concat([first.returns, second.returns]).equals(whole.returns)
    assert pl.concat([first.turnover, second.turnover]).equals(whole.turnover)
    assert pl.concat([first.costs, second.costs]).equals(whole.costs)


def test_resume_carries_unchanged_holdings_without_synthetic_events() -> None:
    times = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    prices = pl.DataFrame(
        {
            "time": times * 3,
            "asset_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "price": [
                10.0,
                11.0,
                12.0,
                13.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                12.0,
                15.0,
                18.0,
            ],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 2 + ["2024-01-03"] * 2,
            "asset_id": ["a", "b", "a", "c"],
            "weight": [0.5, 0.5, 0.5, 0.5],
        }
    )
    config = BacktestConfig(initial_capital=100_000, annualization=4)
    whole = materialize_portfolio_path(
        weights,
        prices,
        identity=_identity(),
        config=config,
    )
    first = materialize_portfolio_path(
        weights.filter(pl.col("time") < "2024-01-03"),
        prices.filter(pl.col("time") <= "2024-01-03"),
        identity=_identity(),
        config=config,
    )
    second = resume_portfolio_path(
        weights.filter(pl.col("time") >= "2024-01-03"),
        prices.filter(pl.col("time") >= "2024-01-03"),
        identity=_identity(),
        checkpoint=first.checkpoint,
        config=config,
    )

    assert pl.concat([first.returns, second.returns]).equals(whole.returns)
    assert pl.concat([first.turnover, second.turnover]).equals(whole.turnover)
    assert pl.concat([first.costs, second.costs]).equals(whole.costs)


def test_result_window_uses_complete_intervals_and_rebases() -> None:
    path = materialize_portfolio_path(
        _weights(),
        _prices(),
        identity=_identity(),
        config=BacktestConfig(initial_capital=100_000, annualization=4),
        execution_availability=_availability(),
    )

    section = compute_result_section(
        path,
        ResultSectionSpec("top_n", ("benchmark_comparison",)),
        ResultWindow(date(2024, 1, 2), date(2024, 1, 4)),
        annualization=4,
    )

    assert section.metrics["benchmark_available"] is False
    assert section.tables["top_n_returns"].get_column("time").to_list() == [
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]
    expected = (
        section.tables["top_n_returns"]
        .select((1.0 + pl.col("net_return")).product() - 1.0)
        .item()
    )
    assert (
        section.tables["top_n_returns"].get_column("net_return_cumulative")[-1]
        == expected
    )


def test_benchmark_metrics_are_recomputed_for_result_window() -> None:
    path = materialize_portfolio_path(
        _weights(),
        _prices(),
        identity=_identity(),
        config=BacktestConfig(initial_capital=100_000, annualization=4),
    )
    benchmark = pl.DataFrame(
        {
            "time": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
            ],
            "benchmark": ["market"] * 4,
            "return": [0.5, 0.01, 0.02, -0.5],
        }
    )
    section = compute_result_section(
        path,
        ResultSectionSpec(
            "top_n",
            ("benchmark_comparison", "excess_return"),
        ),
        ResultWindow(date(2024, 1, 2), date(2024, 1, 4)),
        annualization=4,
        benchmark_returns=benchmark.filter(
            pl.col("time").is_between(date(2024, 1, 2), date(2024, 1, 3))
        ),
    )

    assert section.metrics["benchmark_available"] is True
    assert section.tables["benchmark_returns"]["return"].to_list() == [0.01, 0.02]
    assert section.tables["benchmark_performance"].item(
        0, "total_return"
    ) == pytest.approx(0.0302)
    assert section.tables["excess_returns"]["time"].min() == date(2024, 1, 2)
