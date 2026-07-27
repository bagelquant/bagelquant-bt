from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bagelquant_bt import BacktestConfig, TransactionCostConfig, run_weight_backtest
from bagelquant_bt.exceptions import InputValidationError


def test_weight_backtest_returns_polars_result_frames() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["a", "a", "a"],
            "price": [10.0, 11.0, 12.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["a", "a"],
            "weight": [1.0, 1.0],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=100_000, annualization=252),
    )

    assert result.returns.columns == ["time", "gross_return", "net_return"]
    assert result.value.columns == [
        "time",
        "gross_value",
        "net_value",
        "gross_return_cumulative",
        "net_return_cumulative",
    ]
    assert result.transaction_costs.data["total_fee"].sum() > 0
    assert result.summary.gross_sharpe != result.summary.net_sharpe
    assert result.summary.gross_max_drawdown >= result.summary.net_max_drawdown
    assert result.summary.sharpe == result.summary.net_sharpe
    assert result.summary.max_drawdown == result.summary.net_max_drawdown
    assert result.performance.columns == ["metric", "gross", "net"]
    assert "sharpe" in result.performance["metric"].to_list()


def test_low_frequency_weights_hold_until_next_rebalance() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "asset_id": ["a", "a", "a", "a", "a"],
            "price": [100.0, 110.0, 121.0, 133.1, 146.41],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-03"],
            "asset_id": ["a", "a"],
            "weight": [1.0, 0.5],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(
            initial_capital=10_000,
            annualization=252,
            transaction_cost=TransactionCostConfig(rate=0.001, min_fee=0.0),
        ),
    )

    assert result.weights["weight"].to_list() == [1.0, 1.0, 0.5, 0.5]
    assert result.returns["gross_return"].round(3).to_list() == [0.1, 0.1, 0.05, 0.05]
    assert result.turnover["turnover"].to_list() == [1.0, 0.0, 0.5, 0.0]
    assert result.transaction_costs.data["total_fee"].to_list()[1] == 0.0
    assert result.transaction_costs.data["total_fee"].to_list()[3] == 0.0


def test_gross_and_net_value_paths_follow_independent_daily_ledgers() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["a", "a", "a"],
            "price": [100.0, 110.0, 121.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["a", "a"],
            "weight": [1.0, 1.0],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(
            initial_capital=1_000,
            transaction_cost=TransactionCostConfig(rate=0.01, min_fee=0.0),
        ),
    )

    # Independent hand ledger:
    # gross = 1000 * 1.10 * 1.10
    # net   = (1000 * 1.10 - 10 entry fee) * 1.10
    assert result.value["gross_value"].to_list() == pytest.approx([1_100, 1_210])
    assert result.value["net_value"].to_list() == pytest.approx([1_090, 1_199])
    assert result.value["gross_return_cumulative"].to_list() == pytest.approx(
        [0.10, 0.21]
    )
    assert result.value["net_return_cumulative"].to_list() == pytest.approx(
        [0.09, 0.199]
    )


def test_transaction_cost_min_fee_is_applied_per_traded_asset() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"] * 2,
            "asset_id": ["a", "a", "b", "b"],
            "price": [10.0, 10.0, 20.0, 20.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "weight": [0.5, 0.5],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(
            initial_capital=1_000,
            transaction_cost=TransactionCostConfig(rate=0.001, min_fee=5.0),
        ),
    )

    cost = result.transaction_costs.data.to_dicts()[0]
    assert cost["traded_asset_count"] == 2
    assert cost["traded_notional"] == pytest.approx(1_000.0)
    assert cost["raw_fee"] == pytest.approx(1.0)
    assert cost["total_fee"] == pytest.approx(10.0)
    assert cost["min_fee_adjustment"] == pytest.approx(9.0)


def test_weight_backtest_raises_when_transaction_costs_exhaust_capital() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"] * 3,
            "asset_id": ["a", "a", "b", "b", "c", "c"],
            "price": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b", "c"],
            "weight": [1 / 3, 1 / 3, 1 / 3],
        }
    )

    with pytest.raises(
        InputValidationError,
        match=(
            r"net portfolio value became non-positive.*"
            "Increase initial_capital or reduce traded universe/turnover"
        ),
    ):
        run_weight_backtest(
            weights,
            prices,
            config=BacktestConfig(
                initial_capital=10,
                transaction_cost=TransactionCostConfig(rate=0.001, min_fee=5.0),
            ),
        )


def test_non_price_weight_date_is_dropped() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-03", "2024-01-04"],
            "asset_id": ["a", "a", "a"],
            "price": [10.0, 11.0, 12.0],
        }
    )
    weights = pl.DataFrame({"time": ["2024-01-02"], "asset_id": ["a"], "weight": [1.0]})

    with pytest.raises(InputValidationError, match="at least two overlapping"):
        run_weight_backtest(
            weights,
            prices,
            config=BacktestConfig(initial_capital=10_000),
        )


def test_weight_backtest_drops_missing_price_keys_and_trades_matches() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
            "asset_id": ["a", "a", "a", "b", "b", "b"],
            "price": [10.0, 11.0, 12.0, 20.0, 18.0, 16.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-04",
            ],
            "asset_id": ["a", "b", "c", "a"],
            "weight": [0.5, 0.5, 0.5, 1.0],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=10_000),
    )

    assert set(result.weights["asset_id"]) == {"a", "b"}
    assert result.missing_price_keys.with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d")
    ).to_dicts() == [
        {"time": "2024-01-01", "asset_id": "c"},
        {"time": "2024-01-04", "asset_id": "a"},
    ]
    assert result.returns["gross_return"].round(4).to_list() == [
        0.0,
        pytest.approx(-0.0101),
    ]
    assert result.coverage.with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d")
    ).to_dicts() == [
        {
            "time": "2024-01-01",
            "weight_asset_count": 3,
            "universe_asset_count": 2,
            "coverage_ratio": 1.5,
        },
        {
            "time": "2024-01-02",
            "weight_asset_count": 0,
            "universe_asset_count": 2,
            "coverage_ratio": 0.0,
        },
        {
            "time": "2024-01-03",
            "weight_asset_count": 0,
            "universe_asset_count": 2,
            "coverage_ratio": 0.0,
        },
    ]


def test_weight_backtest_removes_null_and_nan_rows_before_alignment() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["a", "a", "a"],
            "price": [10.0, float("nan"), 12.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["a", "a", "a"],
            "weight": [1.0, None, float("nan")],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=10_000),
    )

    assert result.weights.with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d")
    ).to_dicts() == [{"time": "2024-01-01", "asset_id": "a", "weight": 1.0}]
    assert result.missing_price_keys.is_empty()


def test_missing_asset_price_freezes_daily_mark_and_defers_cumulative_return() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-01",
                "2024-01-04",
            ],
            "asset_id": ["a", "a", "a", "a", "b", "b"],
            "price": [100.0, 110.0, 121.0, 133.1, 100.0, 120.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "weight": [0.5, 0.5],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=10_000),
    )

    assert result.returns["gross_return"].to_list() == pytest.approx([0.05, 0.05, 0.15])
    frozen_returns = result.asset_returns.filter(pl.col("asset_id") == "b")
    frozen_times = frozen_returns.with_columns(pl.col("time").dt.strftime("%Y-%m-%d"))[
        "time"
    ].to_list()
    assert frozen_times == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert frozen_returns["forward_return"].to_list() == pytest.approx([0.0, 0.0, 0.2])
    assert result.price_gaps.with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d"),
        pl.col("last_observed_time").dt.strftime("%Y-%m-%d"),
    ).to_dicts() == [
        {
            "time": "2024-01-02",
            "asset_id": "b",
            "last_observed_time": "2024-01-01",
            "missing_session_count": 1,
        },
        {
            "time": "2024-01-03",
            "asset_id": "b",
            "last_observed_time": "2024-01-01",
            "missing_session_count": 2,
        },
    ]


def test_missing_price_rebalance_keeps_existing_weight_without_trade_cost() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-03",
            ],
            "asset_id": ["a", "a", "a", "b", "b"],
            "price": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "asset_id": ["a", "b", "a"],
            "weight": [0.5, 0.5, 1.0],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(
            initial_capital=10_000,
            transaction_cost=TransactionCostConfig(rate=0.01, min_fee=0.0),
        ),
    )

    assert result.weights.filter(pl.col("time") == pl.date(2024, 1, 2)).sort(
        "asset_id"
    )["weight"].to_list() == [1.0, 0.5]
    assert result.transaction_costs.data["traded_asset_count"].to_list() == [2, 1]
    assert result.unexecuted_weight_keys.with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d")
    ).to_dicts() == [
        {
            "time": "2024-01-02",
            "asset_id": "b",
            "target_weight": 0.0,
            "retained_weight": 0.5,
        }
    ]


def test_missing_price_does_not_open_new_target_weight() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-03",
            ],
            "asset_id": ["a", "a", "a", "b", "b"],
            "price": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["a", "b"],
            "weight": [1.0, 0.5],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=10_000),
    )

    assert result.weights.filter(
        (pl.col("time") == pl.date(2024, 1, 2)) & (pl.col("asset_id") == "b")
    )["weight"].to_list() == [0.0]
    assert result.unexecuted_weight_keys.select(
        "asset_id", "target_weight", "retained_weight"
    ).to_dicts() == [{"asset_id": "b", "target_weight": 0.5, "retained_weight": 0.0}]


def test_trailing_missing_price_keeps_last_valuation_and_is_auditable() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01"],
            "asset_id": ["a", "a", "a", "b"],
            "price": [100.0, 100.0, 100.0, 50.0],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "weight": [0.5, 0.5],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=10_000),
    )

    assert result.returns["gross_return"].to_list() == pytest.approx([0.0, 0.0])
    trailing_gaps = result.price_gaps.filter(pl.col("asset_id") == "b")
    assert trailing_gaps["missing_session_count"].to_list() == [1, 2]


def test_sparse_market_rule_blocks_only_named_buy_and_retries_daily() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            * 2,
            "asset_id": ["cn"] * 4 + ["global"] * 4,
            "price": [100.0, 100.0, 100.0, 110.0] + [100.0] * 4,
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["cn", "global"],
            "weight": [0.5, 0.5],
        }
    )
    availability = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["cn", "cn"],
            "can_buy": [False, False],
            "can_sell": [True, True],
            "reason": ["cn_a_share_price_limit", "cn_a_share_price_limit"],
        }
    )

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(
            initial_capital=10_000,
            transaction_cost=TransactionCostConfig(rate=0.01, min_fee=0.0),
        ),
        execution_availability=availability,
    )

    assert result.weights.sort(["time", "asset_id"]).select(
        "asset_id", "weight"
    ).to_dicts() == [
        {"asset_id": "cn", "weight": 0.0},
        {"asset_id": "global", "weight": 0.5},
        {"asset_id": "cn", "weight": 0.0},
        {"asset_id": "global", "weight": 0.5},
        {"asset_id": "cn", "weight": 0.5},
        {"asset_id": "global", "weight": 0.5},
    ]
    assert result.execution_blocks.select(
        "asset_id", "side", "target_weight", "retained_weight", "reason"
    ).to_dicts() == [
        {
            "asset_id": "cn",
            "side": "buy",
            "target_weight": 0.5,
            "retained_weight": 0.0,
            "reason": "cn_a_share_price_limit",
        },
        {
            "asset_id": "cn",
            "side": "buy",
            "target_weight": 0.5,
            "retained_weight": 0.0,
            "reason": "cn_a_share_price_limit",
        },
    ]
    assert result.returns["gross_return"].to_list() == pytest.approx(
        [0.0, 0.0, 0.05]
    )
    assert result.transaction_costs.data.select(
        pl.col("time").dt.strftime("%Y-%m-%d"), "traded_asset_count"
    ).to_dicts() == [
        {"time": "2024-01-01", "traded_asset_count": 1},
        {"time": "2024-01-02", "traded_asset_count": 0},
        {"time": "2024-01-03", "traded_asset_count": 1},
    ]
    assert result.turnover.get_column("turnover").to_list() == [0.5, 0.0, 0.5]


def test_buy_constraint_does_not_block_sell_and_new_target_supersedes_pending() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["cn"] * 3,
            "price": [100.0, 100.0, 100.0],
        }
    )
    availability = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["cn", "cn"],
            "can_buy": [False, False],
            "can_sell": [True, True],
            "reason": ["limit", "limit"],
        }
    )
    pending_then_cancelled = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["cn", "cn"],
            "weight": [1.0, 0.0],
        }
    )
    cancelled = run_weight_backtest(
        pending_then_cancelled,
        prices,
        config=BacktestConfig(initial_capital=10_000),
        execution_availability=availability,
    )
    assert cancelled.weights["weight"].to_list() == [0.0, 0.0]
    assert cancelled.execution_blocks.height == 1

    held_then_sold = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["cn", "cn"],
            "weight": [1.0, 0.0],
        }
    )
    sell_only_constraint = availability.filter(
        pl.col("time").str.to_date() == date(2024, 1, 2)
    )
    sold = run_weight_backtest(
        held_then_sold,
        prices,
        config=BacktestConfig(initial_capital=10_000),
        execution_availability=sell_only_constraint,
    )
    assert sold.weights["weight"].to_list() == [1.0, 0.0]
    assert sold.execution_blocks.is_empty()

    blocked_sell = run_weight_backtest(
        held_then_sold,
        prices,
        config=BacktestConfig(initial_capital=10_000),
        execution_availability=pl.DataFrame(
            {
                "time": ["2024-01-02"],
                "asset_id": ["cn"],
                "can_buy": [True],
                "can_sell": [False],
                "reason": ["explicit_sell_block"],
            }
        ),
    )
    assert blocked_sell.weights["weight"].to_list() == [1.0, 1.0]
    assert blocked_sell.execution_blocks.select("side", "reason").to_dicts() == [
        {"side": "sell", "reason": "explicit_sell_block"}
    ]
