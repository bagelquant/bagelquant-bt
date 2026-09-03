from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from bagelquant_bt import (
    AccountBacktestConfig,
    InputValidationError,
    TransactionCostConfig,
    run_account_backtest,
    run_planned_account_backtest,
    run_stateful_account_backtest,
)


def _coverage(*days: date) -> pl.DataFrame:
    return pl.DataFrame({"time": list(days), "is_complete": [True] * len(days)})


def _zero_cost() -> TransactionCostConfig:
    return TransactionCostConfig(
        rate=0.0,
        min_fee=0.0,
        buy_slippage_rate=0.0,
        sell_slippage_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
    )


def test_account_buys_and_sells_whole_lots_with_t_plus_one() -> None:
    days = [date(2024, 1, day) for day in range(2, 5)]
    prices = pl.DataFrame(
        {
            "time": days,
            "asset_id": ["a"] * 3,
            "open": [100.0] * 3,
            "close": [100.0] * 3,
        }
    )
    targets = pl.DataFrame(
        {
            "time": [days[0], days[1]],
            "asset_id": ["a", "a"],
            "weight": [1.0, 0.0],
        }
    )

    result = run_account_backtest(
        targets,
        prices,
        corporate_action_coverage=_coverage(*days),
        lot_sizes=pl.DataFrame({"asset_id": ["a"], "buy_lot_size": [100]}),
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=10_500.0,
            settlement_sessions=1,
            transaction_cost=_zero_cost(),
        ),
    )

    assert result.fills.select("side", "quantity").to_dicts() == [
        {"side": "buy", "quantity": 100},
        {"side": "sell", "quantity": 100},
    ]
    assert result.account_value.get_column("equity").to_list() == pytest.approx(
        [10_500.0, 10_500.0, 10_500.0]
    )
    assert result.final_checkpoint.cash == pytest.approx(10_500.0)


def test_planned_account_executes_frozen_quantity_and_expires_remainder() -> None:
    decision = date(2024, 8, 30)
    execution = date(2024, 9, 2)
    following = date(2024, 9, 3)
    plans = pl.DataFrame(
        {
            "decision_date": [decision, decision],
            "execution_date": [execution, execution],
            "asset_id": ["a", "b"],
            "target_weight": [0.5, 0.5],
            "sizing_notional": [10_000.0, 10_000.0],
            "decision_price": [10.0, 10.0],
            "target_quantity": [500, 500],
        }
    )
    prices = pl.DataFrame(
        {
            "time": [execution, execution, following, following],
            "asset_id": ["a", "b", "a", "b"],
            "open": [20.0, 10.0, 10.0, 10.0],
            "close": [20.0, 10.0, 10.0, 10.0],
        }
    )
    availability = pl.DataFrame(
        {
            "time": [execution, execution, following, following],
            "asset_id": ["a", "b", "a", "b"],
            "can_buy": [True, False, True, True],
            "can_sell": [True, True, True, True],
            "reason": [None, "limit_up", None, None],
        }
    )

    result = run_planned_account_backtest(
        plans,
        prices,
        corporate_action_coverage=_coverage(execution, following),
        execution_availability=availability,
        lot_sizes=pl.DataFrame(
            {"asset_id": ["a", "b"], "buy_lot_size": [100, 100]}
        ),
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=10_000.0,
            retry_blocked_orders=True,
            transaction_cost=_zero_cost(),
        ),
    )

    assert result.target_positions.select(
        "asset_id", "decision_price", "open_price", "target_quantity"
    ).to_dicts() == [
        {
            "asset_id": "a",
            "decision_price": 10.0,
            "open_price": 20.0,
            "target_quantity": 500,
        },
        {
            "asset_id": "b",
            "decision_price": 10.0,
            "open_price": 10.0,
            "target_quantity": 500,
        },
    ]
    assert result.fills.select("asset_id", "quantity").to_dicts() == [
        {"asset_id": "a", "quantity": 500}
    ]
    assert result.orders.filter(pl.col("asset_id") == "b").select(
        "time",
        "requested_quantity",
        "filled_quantity",
        "unfilled_quantity",
        "implementation_gap",
        "status",
        "expires_at",
    ).to_dicts() == [
        {
            "time": execution,
            "requested_quantity": 500,
            "filled_quantity": 0,
            "unfilled_quantity": 500,
            "implementation_gap": 1.0,
            "status": "expired",
            "expires_at": execution,
        }
    ]
    assert result.final_checkpoint.pending_target_positions.is_empty()
    assert result.executable_weights.filter(pl.col("time") == execution).item(
        0, "actual_weight"
    ) == pytest.approx(1.0)


def test_stateful_account_sizes_and_executes_all_decisions_in_one_pass() -> None:
    days = [date(2024, 8, 30), date(2024, 9, 2), date(2024, 9, 3)]
    calls = []

    def targets(context):
        calls.append(context)
        weight = 1.0 if context.decision_date == days[0] else 0.0
        return pl.DataFrame({"asset_id": ["a"], "weight": [weight]})

    result = run_stateful_account_backtest(
        pl.DataFrame(
            {
                "decision_date": days[:2],
                "execution_date": days[1:],
            }
        ),
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a"] * 3,
                "open": [10.0, 20.0, 20.0],
                "close": [10.0, 20.0, 20.0],
            }
        ),
        targets,
        corporate_action_coverage=_coverage(*days),
        lot_sizes=pl.DataFrame({"asset_id": ["a"], "buy_lot_size": [1]}),
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=1_000.0,
            transaction_cost=_zero_cost(),
        ),
    )

    assert [item.decision_date for item in calls] == days[:2]
    assert calls[1].reference_weights.select("asset_id", "weight").to_dicts() == [
        {"asset_id": "a", "weight": 1.0}
    ]
    assert result.target_position_plans.select(
        "decision_date", "execution_date", "target_quantity"
    ).to_dicts() == [
        {
            "decision_date": days[0],
            "execution_date": days[1],
            "target_quantity": 100,
        },
        {
            "decision_date": days[1],
            "execution_date": days[2],
            "target_quantity": 0,
        },
    ]
    assert result.account.fills.select("time", "side", "quantity").to_dicts() == [
        {"time": days[1], "side": "buy", "quantity": 50},
        {"time": days[2], "side": "sell", "quantity": 50},
    ]
    buy = result.account.orders.filter(pl.col("side") == "buy").row(0, named=True)
    assert buy["requested_quantity"] == 100
    assert buy["unfilled_quantity"] == 50
    assert buy["expires_at"] == days[1]


def test_stateful_account_freezes_unpriced_holding_outside_decision_plan() -> None:
    decision = date(2024, 8, 30)
    execution = date(2024, 9, 2)
    result = run_stateful_account_backtest(
        pl.DataFrame(
            {"decision_date": [decision], "execution_date": [execution]}
        ),
        pl.DataFrame(
            {
                "time": [decision, execution],
                "asset_id": ["b", "b"],
                "open": [10.0, 10.0],
                "close": [10.0, 10.0],
            }
        ),
        lambda _context: pl.DataFrame({"asset_id": ["b"], "weight": [1.0]}),
        corporate_action_coverage=_coverage(decision, execution),
        initial_positions=pl.DataFrame(
            {
                "asset_id": ["a"],
                "quantity": [100],
                "available_quantity": [100],
                "last_mark": [10.0],
            }
        ),
        initial_cash=0.0,
        config=AccountBacktestConfig(
            capital_mode="compounding",
            transaction_cost=_zero_cost(),
        ),
    )

    assert result.target_position_plans.select(
        "asset_id", "target_weight", "decision_price", "target_quantity"
    ).to_dicts() == [
        {
            "asset_id": "b",
            "target_weight": 1.0,
            "decision_price": 10.0,
            "target_quantity": 100,
        }
    ]
    assert result.account.fills.is_empty()
    assert result.account.final_checkpoint.positions.select(
        "asset_id", "quantity", "last_mark"
    ).to_dicts() == [
        {"asset_id": "a", "quantity": 100, "last_mark": 10.0}
    ]


def test_stateful_account_rejects_unpriced_target_asset() -> None:
    decision = date(2024, 8, 30)
    execution = date(2024, 9, 2)

    with pytest.raises(
        InputValidationError,
        match="decision sizing requires a finite close for target assets: b",
    ):
        run_stateful_account_backtest(
            pl.DataFrame(
                {"decision_date": [decision], "execution_date": [execution]}
            ),
            pl.DataFrame(
                {
                    "time": [decision, execution],
                    "asset_id": ["a", "a"],
                    "open": [10.0, 10.0],
                    "close": [10.0, 10.0],
                }
            ),
            lambda _context: pl.DataFrame(
                {"asset_id": ["b"], "weight": [1.0]}
            ),
            corporate_action_coverage=_coverage(decision, execution),
            config=AccountBacktestConfig(
                capital_mode="compounding",
                initial_capital=1_000.0,
                transaction_cost=_zero_cost(),
            ),
        )


def test_stateful_account_resumes_checkpoint_and_keeps_prior_plans() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    prices = pl.DataFrame(
        {
            "time": days,
            "asset_id": ["a"] * 3,
            "open": [10.0] * 3,
            "close": [10.0] * 3,
        }
    )
    config = AccountBacktestConfig(
        capital_mode="compounding",
        initial_capital=1_000.0,
        transaction_cost=_zero_cost(),
    )
    first = run_stateful_account_backtest(
        pl.DataFrame(
            {"decision_date": [days[0]], "execution_date": [days[1]]}
        ),
        prices.filter(pl.col("time") <= days[1]),
        lambda _context: pl.DataFrame({"asset_id": ["a"], "weight": [1.0]}),
        corporate_action_coverage=_coverage(*days[:2]),
        config=config,
    )
    resumed = run_stateful_account_backtest(
        pl.DataFrame(
            {"decision_date": [days[1]], "execution_date": [days[2]]}
        ),
        prices.filter(pl.col("time") >= days[1]),
        lambda _context: pl.DataFrame({"asset_id": ["a"], "weight": [0.0]}),
        corporate_action_coverage=_coverage(*days[1:]),
        config=config,
        checkpoint=first.account.final_checkpoint,
        initial_target_position_plans=first.target_position_plans,
    )

    assert resumed.target_position_plans.get_column(
        "decision_date"
    ).to_list() == days[:2]
    assert resumed.account.fills.select("time", "side", "quantity").to_dicts() == [
        {"time": days[2], "side": "sell", "quantity": 100}
    ]


def test_t_plus_one_pending_sell_has_reason_and_retries_once_available() -> None:
    days = [date(2024, 1, day) for day in range(2, 6)]
    result = run_account_backtest(
        pl.DataFrame(
            {
                "time": [days[0], days[1]],
                "asset_id": ["a", "a"],
                "weight": [1.0, 0.0],
            }
        ),
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a"] * len(days),
                "open": [100.0] * len(days),
                "close": [100.0] * len(days),
            }
        ),
        corporate_action_coverage=_coverage(*days),
        lot_sizes=pl.DataFrame({"asset_id": ["a"], "buy_lot_size": [100]}),
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=10_500.0,
            settlement_sessions=2,
            transaction_cost=_zero_cost(),
        ),
    )

    assert result.orders.select(
        "time", "side", "order_quantity", "status", "reason"
    ).to_dicts() == [
        {
            "time": days[0],
            "side": "buy",
            "order_quantity": 100,
            "status": "filled",
            "reason": None,
        },
        {
            "time": days[1],
            "side": "sell",
            "order_quantity": 0,
            "status": "pending",
            "reason": "t_plus_one_unavailable",
        },
        {
            "time": days[2],
            "side": "sell",
            "order_quantity": 100,
            "status": "filled",
            "reason": None,
        },
    ]


def test_account_leaves_unaffordable_high_price_lot_pending() -> None:
    day = date(2024, 1, 2)
    result = run_account_backtest(
        pl.DataFrame({"time": [day], "asset_id": ["a"], "weight": [1.0]}),
        pl.DataFrame(
            {
                "time": [day],
                "asset_id": ["a"],
                "open": [6_000.0],
                "close": [6_000.0],
            }
        ),
        corporate_action_coverage=_coverage(day),
        lot_sizes=pl.DataFrame({"asset_id": ["a"], "buy_lot_size": [100]}),
        config=AccountBacktestConfig(transaction_cost=_zero_cost()),
    )

    assert result.fills.is_empty()
    assert result.orders.is_empty()
    assert result.target_positions.get_column("target_quantity").to_list() == [0]


def test_late_order_reason_does_not_depend_on_schema_inference_window() -> None:
    days = [date(2024, 1, 2) + timedelta(days=index) for index in range(120)]
    result = run_account_backtest(
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a"] * len(days),
                "weight": [1.0 if index % 2 == 0 else 0.0 for index in range(120)],
            }
        ),
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a"] * len(days),
                "open": [10.0] * len(days),
                "close": [10.0] * len(days),
            }
        ),
        corporate_action_coverage=_coverage(*days),
        execution_availability=pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a"] * len(days),
                "can_buy": [True] * 110 + [False] * 10,
                "can_sell": [True] * len(days),
                "reason": [None] * 110 + ["cn_a_share_price_limits:limit_up"] * 10,
            }
        ),
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=1_000.0,
            transaction_cost=_zero_cost(),
        ),
    )

    assert result.orders.schema["reason"] == pl.String
    assert (
        result.orders.get_column("reason").drop_nulls().tail(1).item()
        == "cn_a_share_price_limits:limit_up"
    )


def test_account_applies_distinct_buy_and_sell_slippage_rates() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    result = run_account_backtest(
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a", "a"],
                "weight": [1.0, 0.0],
            }
        ),
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a", "a"],
                "open": [10.0, 10.0],
                "close": [10.0, 10.0],
            }
        ),
        corporate_action_coverage=_coverage(*days),
        lot_sizes=pl.DataFrame({"asset_id": ["a"], "buy_lot_size": [100]}),
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=1_010.0,
            transaction_cost=TransactionCostConfig(
                rate=0.0,
                min_fee=0.0,
                buy_slippage_rate=0.01,
                sell_slippage_rate=0.02,
                stamp_tax_rate=0.0,
                transfer_fee_rate=0.0,
            ),
        ),
    )

    assert result.fills.select("side", "fill_price").to_dicts() == [
        {"side": "buy", "fill_price": pytest.approx(10.1)},
        {"side": "sell", "fill_price": pytest.approx(9.8)},
    ]


def test_fixed_notional_withdrawal_preserves_unit_nav() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    result = run_account_backtest(
        pl.DataFrame(
            {
                "time": [days[0], days[2]],
                "asset_id": ["a", "a"],
                "weight": [1.0, 1.0],
            }
        ),
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a", "a", "a"],
                "open": [100.0, 110.0, 110.0],
                "close": [110.0, 110.0, 110.0],
            }
        ),
        corporate_action_coverage=_coverage(*days),
        lot_sizes=pl.DataFrame({"asset_id": ["a"], "buy_lot_size": [100]}),
        config=AccountBacktestConfig(
            settlement_sessions=0,
            transaction_cost=_zero_cost(),
        ),
    )

    assert result.external_flows.select("amount", "reason").to_dicts() == [
        {"amount": -50_000.0, "reason": "fixed_notional_withdrawal"}
    ]
    assert result.account_value.get_column("equity").to_list() == pytest.approx(
        [550_000.0, 550_000.0, 500_000.0]
    )
    assert result.account_value.get_column("nav").to_list() == pytest.approx(
        [1.1, 1.1, 1.1]
    )


def test_compounding_rebalances_only_on_target_sessions_using_full_equity() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    result = run_account_backtest(
        pl.DataFrame(
            {
                "time": [days[0], days[2]],
                "asset_id": ["a", "a"],
                "weight": [1.0, 1.0],
            }
        ),
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a", "a", "a"],
                "open": [100.0, 110.0, 110.0],
                "close": [110.0, 110.0, 110.0],
            }
        ),
        corporate_action_coverage=_coverage(*days),
        lot_sizes=pl.DataFrame({"asset_id": ["a"], "buy_lot_size": [100]}),
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=500_000.0,
            transaction_cost=_zero_cost(),
        ),
    )

    assert result.external_flows.is_empty()
    assert result.orders.get_column("time").unique().sort().to_list() == [days[0]]
    assert result.positions.filter(pl.col("time") == days[1]).get_column(
        "quantity"
    ).to_list() == [5_000]
    assert result.target_positions.filter(pl.col("time") == days[2]).get_column(
        "sizing_capital"
    ).to_list() == pytest.approx([550_000.0])


def test_blocked_order_retries_without_rebalancing_filled_assets() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    result = run_account_backtest(
        pl.DataFrame(
            {
                "time": [days[0], days[0]],
                "asset_id": ["a", "b"],
                "weight": [0.5, 0.5],
            }
        ),
        pl.DataFrame(
            {
                "time": days * 2,
                "asset_id": ["a"] * 3 + ["b"] * 3,
                "open": [10.0] * 6,
                "close": [10.0] * 6,
            }
        ).sort("time", "asset_id"),
        corporate_action_coverage=_coverage(*days),
        execution_availability=pl.DataFrame(
            {
                "time": days * 2,
                "asset_id": ["a"] * 3 + ["b"] * 3,
                "can_buy": [True, True, True, False, True, True],
                "can_sell": [True] * 6,
                "reason": [None, None, None, "blocked", None, None],
            }
        ).sort("time", "asset_id"),
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=10_000.0,
            transaction_cost=_zero_cost(),
        ),
    )

    orders = result.orders.select("time", "asset_id", "status").to_dicts()
    assert orders == [
        {"time": days[0], "asset_id": "a", "status": "filled"},
        {"time": days[0], "asset_id": "b", "status": "pending"},
        {"time": days[1], "asset_id": "b", "status": "filled"},
    ]


def test_cash_dividend_moves_from_receivable_to_cash() -> None:
    days = [date(2024, 1, day) for day in range(2, 5)]
    actions = pl.DataFrame(
        {
            "action_id": ["div-a"],
            "asset_id": ["a"],
            "is_implemented": [True],
            "record_date": [days[0]],
            "ex_date": [days[1]],
            "cash_pay_date": [days[2]],
            "share_available_date": [None],
            "cash_dividend_per_share": [0.1],
            "stock_dividend_per_share": [0.0],
        }
    )
    result = run_account_backtest(
        pl.DataFrame({"time": [days[0]], "asset_id": ["a"], "weight": [1.0]}),
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a"] * 3,
                "open": [10.0, 9.9, 9.9],
                "close": [10.0, 9.9, 9.9],
            }
        ),
        corporate_action_coverage=_coverage(*days),
        corporate_actions=actions,
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=10_000.0,
            transaction_cost=_zero_cost(),
        ),
    )

    assert result.receivables.select("kind", "amount").to_dicts() == [
        {"kind": "cash_created", "amount": 100.0},
        {"kind": "cash_paid", "amount": 100.0},
    ]
    assert result.account_value.get_column("equity").to_list() == pytest.approx(
        [10_000.0, 10_000.0, 10_000.0]
    )


def test_checkpoint_resume_matches_full_account_values() -> None:
    days = [date(2024, 1, day) for day in range(2, 6)]
    targets = pl.DataFrame({"time": [days[0]], "asset_id": ["a"], "weight": [1.0]})
    prices = pl.DataFrame(
        {
            "time": days,
            "asset_id": ["a"] * 4,
            "open": [10.0, 11.0, 12.0, 13.0],
            "close": [11.0, 12.0, 13.0, 14.0],
        }
    )
    config = AccountBacktestConfig(
        capital_mode="compounding",
        initial_capital=10_000.0,
        transaction_cost=_zero_cost(),
    )
    full = run_account_backtest(
        targets,
        prices,
        corporate_action_coverage=_coverage(*days),
        config=config,
    )
    prefix = run_account_backtest(
        targets,
        prices.filter(pl.col("time") <= days[1]),
        corporate_action_coverage=_coverage(*days[:2]),
        config=config,
    )
    resumed = run_account_backtest(
        targets.clear(),
        prices.filter(pl.col("time") > days[1]),
        corporate_action_coverage=_coverage(*days[2:]),
        config=config,
        checkpoint=prefix.final_checkpoint,
    )

    assert (
        resumed.account_value.select("time", "equity", "nav").to_dicts()
        == full.account_value.filter(pl.col("time") > days[1])
        .select("time", "equity", "nav")
        .to_dicts()
    )
