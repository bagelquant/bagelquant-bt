from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from bagelquant_bt import (
    RESULT_SECTION_VERSION,
    BacktestConfig,
    PortfolioPathIdentity,
    ResultSectionSpec,
    ResultWindow,
    TransactionCostConfig,
    compute_result_section,
    materialize_portfolio_path,
    resume_portfolio_path,
)
from bagelquant_bt.path import PORTFOLIO_PATH_VERSION
from bagelquant_bt.window import compute_window_tables


def test_result_section_version_is_public() -> None:
    assert PORTFOLIO_PATH_VERSION == 3
    assert RESULT_SECTION_VERSION == 9


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


def test_resumed_path_preserves_bankruptcy_state() -> None:
    times = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    prices = pl.DataFrame(
        {
            "time": times * 3,
            "asset_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "price": [10.0] * 4 + [20.0] * 4 + [30.0] * 4,
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01"] * 3 + ["2024-01-03"] * 3,
            "asset_id": ["a", "b", "c"] * 2,
            "weight": [1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 1.0],
        }
    )
    config = BacktestConfig(
        initial_capital=10,
        transaction_cost=TransactionCostConfig(
            rate=0.001,
            min_fee=5.0,
            buy_slippage_rate=0.0,
            sell_slippage_rate=0.0,
            stamp_tax_rate=0.0,
        ),
        insolvency_action="freeze_zero",
    )
    whole = materialize_portfolio_path(
        weights,
        prices,
        identity=_identity(),
        config=config,
    )
    first = materialize_portfolio_path(
        weights.filter(pl.col("time") == "2024-01-01"),
        prices.filter(pl.col("time") <= "2024-01-02"),
        identity=_identity(),
        config=config,
    )
    second = resume_portfolio_path(
        weights.filter(pl.col("time") >= "2024-01-03"),
        prices.filter(pl.col("time") >= "2024-01-02"),
        identity=_identity(),
        checkpoint=first.checkpoint,
        config=config,
    )

    assert first.checkpoint.is_bankrupt is True
    assert pl.concat([first.returns, second.returns]).equals(whole.returns)
    assert pl.concat([first.turnover, second.turnover]).equals(whole.turnover)
    assert pl.concat([first.costs, second.costs]).equals(whole.costs)
    assert second.checkpoint.is_bankrupt is True


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


def test_ic_section_accepts_preaggregated_decay_table() -> None:
    path = materialize_portfolio_path(
        _weights(),
        _prices(),
        identity=_identity(),
        config=BacktestConfig(initial_capital=100_000, annualization=4),
    )
    path = replace(
        path,
        series={
            **path.series,
            "ic_decay": pl.DataFrame(
                {
                    "lag": [0, 1],
                    "method": ["spearman", "spearman"],
                    "ic_mean": [0.10, 0.08],
                }
            ),
        },
    )

    section = compute_result_section(
        path,
        ResultSectionSpec("ic", ("ic_decay",)),
        ResultWindow(date(2024, 1, 2), date(2024, 1, 4)),
        annualization=4,
    )

    assert section.tables["ic_decay"].to_dicts() == [
        {"lag": 0, "method": "spearman", "ic_mean": 0.10},
        {"lag": 1, "method": "spearman", "ic_mean": 0.08},
    ]


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


def test_window_sections_preserve_bankruptcy_markers_and_outputs() -> None:
    times = [date(2024, 1, day) for day in range(2, 5)]
    returns = pl.DataFrame(
        {
            "time": times,
            "gross_return": [0.0, 0.0, 0.0],
            "net_return": [-1.0, 0.0, 0.0],
            "is_bankrupt": [True, True, True],
            "bankruptcy_event": [True, False, False],
        }
    )
    lag_returns = pl.DataFrame(
        {
            "portfolio": ["spread"] * 3 + ["spread"] * 3 + ["top_n"] * 3,
            "lag": [0] * 3 + [1] * 3 + [0] * 3,
            "time": times * 3,
            "gross_return": [0.0] * 6 + [0.01, 0.0, 0.0],
            "net_return": [-1.0, 0.0, 0.0] + [0.01, 0.0, 0.0] * 2,
            "is_bankrupt": [True, True, True] + [False] * 6,
            "bankruptcy_event": [True, False, False] + [False] * 6,
        }
    )
    quantile_returns = pl.DataFrame(
        {
            "quantile": ["q1"] * 3 + ["q2"] * 3,
            "time": times * 2,
            "return": [-1.0, 0.0, 0.0, 0.01, 0.0, 0.0],
            "is_bankrupt": [True, True, True, False, False, False],
            "bankruptcy_event": [True, False, False, False, False, False],
        }
    )
    series = {
        "factor": pl.DataFrame({"time": [date(2024, 1, 1), *times]}),
        "lag_returns": lag_returns,
        "quantile_returns": quantile_returns,
    }
    turnover = pl.DataFrame({"time": times, "turnover": [1.0, 0.0, 0.0]})

    summary_metrics, summary_tables = compute_window_tables(
        "summary",
        (),
        returns=returns,
        turnover=turnover,
        costs=pl.DataFrame(),
        series=series,
        annualization=3,
        ic_annualization=3,
    )
    assert summary_metrics["top_n_is_bankrupt"] is True
    assert summary_metrics["spread_is_bankrupt"] is True
    assert summary_metrics["top_n_net_annualized_return"] == -1.0
    assert summary_tables["bankruptcies"].height == 2

    _, spread_tables = compute_window_tables(
        "spread",
        ("spread_time_series", "spread_lag_performance"),
        returns=returns,
        turnover=turnover,
        costs=pl.DataFrame(),
        series=series,
        annualization=3,
        ic_annualization=3,
    )
    assert spread_tables["spread_lag_performance"].filter(
        pl.col("lag") == 0
    ).item(0, "bankruptcy_time") == times[0]
    assert spread_tables["bankruptcies"].height == 1

    _, top_n_tables = compute_window_tables(
        "top_n",
        ("benchmark_comparison", "lag_performance"),
        returns=returns,
        turnover=turnover,
        costs=pl.DataFrame(),
        series=series,
        annualization=3,
        ic_annualization=3,
    )
    assert top_n_tables["top_n_returns"]["net_return_cumulative"].to_list() == [
        -1.0,
        -1.0,
        -1.0,
    ]
    assert top_n_tables["bankruptcies"].height == 1

    _, quantile_tables = compute_window_tables(
        "quantiles",
        ("annualized_return", "time_series"),
        returns=returns,
        turnover=turnover,
        costs=pl.DataFrame(),
        series=series,
        annualization=3,
        ic_annualization=3,
    )
    q1 = quantile_tables["quantile_performance"].filter(pl.col("quantile") == "q1")
    assert q1.item(0, "is_bankrupt") is True
    assert q1.item(0, "bankruptcy_time") == times[0]
    assert quantile_tables["bankruptcies"].height == 1

    _, statistics_tables = compute_window_tables(
        "statistical_tests",
        (),
        returns=returns,
        turnover=turnover,
        costs=pl.DataFrame(),
        series=series,
        annualization=3,
        ic_annualization=3,
    )
    assert statistics_tables["statistical_tests"].get_column("test").to_list() == [
        "pearson_ic",
        "spearman_ic",
        "quantile_rank_ic",
        "spread_net_return",
        "cross_section_regression",
    ]
    assert statistics_tables["bankruptcies"].height == 1


def test_result_statistics_share_complete_execution_period_sample() -> None:
    path = materialize_portfolio_path(
        _weights(),
        _prices(),
        identity=_identity(),
        config=BacktestConfig(initial_capital=100_000, annualization=4),
    )
    schedule = [date(2024, 1, 1), date(2024, 1, 3), date(2024, 1, 5)]
    quantile_rows = []
    for day in [date(2024, 1, value) for value in range(1, 5)]:
        quantile_rows.extend(
            [
                {"time": day, "quantile": "q1", "return": 0.01},
                {"time": day, "quantile": "q2", "return": -0.01},
            ]
        )
    path = replace(
        path,
        series={
            "factor": pl.DataFrame({"time": schedule}),
            "ic": pl.DataFrame(
                {
                    "time": schedule,
                    "pearson_ic": [0.1, 0.2, 0.9],
                    "spearman_ic": [0.2, 0.3, 0.9],
                }
            ),
            "quantile_returns": pl.DataFrame(quantile_rows),
        },
    )

    section = compute_result_section(
        path,
        ResultSectionSpec("statistical_tests"),
        ResultWindow(schedule[0], schedule[-1]),
        annualization=4,
    )
    samples = {
        row["test"]: row["sample_size"]
        for row in section.tables["statistical_tests"].iter_rows(named=True)
    }

    assert samples["pearson_ic"] == 2
    assert samples["spearman_ic"] == 2
    assert samples["quantile_rank_ic"] == 2


def test_quantile_window_tables_use_numeric_label_order() -> None:
    labels = ["q1", "q10", *[f"q{number}" for number in range(2, 10)]]
    returns = pl.DataFrame(
        {
            "time": [date(2024, 1, 2)] * 10,
            "quantile": labels,
            "return": [number / 100 for number in range(10)],
        }
    )

    _, tables = compute_window_tables(
        "quantiles",
        ("annualized_return", "time_series"),
        returns=pl.DataFrame(),
        turnover=pl.DataFrame(),
        costs=pl.DataFrame(),
        series={"quantile_returns": returns},
        annualization=12,
        ic_annualization=12,
    )
    expected = [f"q{number}" for number in range(1, 11)]

    assert tables["quantile_performance"].get_column("quantile").to_list() == expected
    assert (
        tables["quantile_returns"]
        .get_column("quantile")
        .unique(maintain_order=True)
        .to_list()
        == expected
    )
