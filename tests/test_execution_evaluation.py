"""Hand-checkable signed-delta, account and FIFO contracts."""

from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt.exceptions import InputValidationError
from bagelquant_bt.execution_evaluation import (
    ExecutionConfig,
    ExecutionCostQuote,
    ExecutionCostRule,
    evaluate_execution,
    execution_stress_scenarios,
    load_checkpoint,
    logical_content_hash,
    merge_execution_tables,
    run_execution_from_weights,
    save_checkpoint,
    summarize_transaction_pnl,
)

D1 = date(2024, 1, 2)
D2 = D1 + timedelta(days=1)
D3 = D2 + timedelta(days=1)
D4 = D3 + timedelta(days=1)


def market(rows):
    return pl.DataFrame(
        rows,
        schema={
            "time": pl.Date,
            "asset_id": pl.String,
            "execution_price": pl.Float64,
            "valuation_price": pl.Float64,
        },
        orient="row",
    )


def plans(rows):
    return pl.DataFrame(
        rows,
        schema={
            "plan_id": pl.String,
            "execution_date": pl.Date,
            "asset_id": pl.String,
            "quantity": pl.Int64,
        },
        orient="row",
    )


def test_default_partial_fill_minimum_fee_and_fifo_reconcile():
    # Ideal buys 10 x 10 and sells all at 12. Actual can afford 9 shares
    # plus the minimum fee. Actual sells nine, paying a second minimum fee.
    result = evaluate_execution(
        plans([("buy", D1, "a", 10), ("sell", D2, "a", -10)]),
        market([(D1, "a", 10, 10), (D2, "a", 12, 12)]),
        initial_capital=100,
    )
    assert result.tables["fills"]["quantity"].to_list() == [9, 9]
    assert result.tables["fills"]["commission"].to_list() == [5, 5]
    assert result.tables["orders"]["unfilled_quantity"].to_list() == [1, 1]
    assert result.tables["account_value"]["equity"].to_list() == [95, 108]
    assert result.tables["account_value"]["target_equity"].to_list() == [100, 120]
    match = (
        result.tables["fifo_matches"]
        .filter(pl.col("ledger") == "actual")
        .row(0, named=True)
    )
    assert match["gross_pnl"] == 18
    assert match["entry_cost"] == match["exit_cost"] == 5
    assert match["net_pnl"] == 8
    assert result.metrics["actual_pnl"] == 8
    assert result.metrics["target_pnl"] == 20
    assert result.tables["account_value"]["reconciliation_error"].abs().max() < 1e-9


@pytest.mark.parametrize(
    "rows,match",
    [
        ([("buy", D1, "a", 11)], "cash-infeasible"),
        ([("sell", D1, "a", -1)], "short inventory"),
    ],
)
def test_infeasible_ideal_plan_rejects_entire_evaluation(rows, match):
    with pytest.raises(InputValidationError, match=match):
        evaluate_execution(
            plans(rows), market([(D1, "a", 10, 10)]), initial_capital=100
        )


def test_frozen_deltas_are_not_resized_or_netted_and_sell_first():
    transactions = plans(
        [
            ("first", D1, "z", 10),
            ("buy_a", D2, "a", 10),
            ("sell_z", D2, "z", -10),
        ]
    )
    prices = market([(D1, "z", 10, 10), (D2, "a", 10, 10), (D2, "z", 10, 10)])
    result = evaluate_execution(
        transactions,
        prices,
        initial_capital=100,
        config=ExecutionConfig(min_fee=0, rate=0),
    )
    assert result.tables["fills"]["plan_id"].to_list() == ["first", "sell_z", "buy_a"]
    assert result.tables["fills"]["quantity"].to_list() == [10, 10, 10]
    separate = evaluate_execution(
        plans([("a1", D1, "z", 2), ("a2", D1, "z", 2)]), prices, initial_capital=100
    )
    assert separate.tables["fills"]["commission"].to_list() == [5, 5]


def test_fifo_partial_sale_allocates_costs_and_keeps_open_lot_basis():
    result = evaluate_execution(
        plans(
            [
                ("b1", D1, "a", 4),
                ("b2", D2, "a", 3),
                ("s1", D3, "a", -5),
            ]
        ),
        market([(D1, "a", 10, 10), (D2, "a", 20, 20), (D3, "a", 30, 30)]),
        initial_capital=200,
    )
    matches = result.tables["fifo_matches"].filter(pl.col("ledger") == "actual")
    assert matches["buy_plan_id"].to_list() == ["b1", "b2"]
    assert matches["quantity"].to_list() == [4, 1]
    assert matches["entry_cost"].to_list() == pytest.approx([5, 5 / 3])
    assert matches["exit_cost"].to_list() == [4, 1]
    remaining = result.tables["open_lots"].row(0, named=True)
    assert remaining["quantity"] == 2
    assert remaining["entry_fee"] == pytest.approx(10 / 3)
    assert result.tables["account_value"]["equity"][-1] == 295


def test_blocked_order_expires_by_default_and_explicit_retry_uses_residual():
    prices = market([(D1, "a", 10, 10), (D2, "a", 11, 11)])
    availability = pl.DataFrame(
        {
            "time": [D1],
            "asset_id": ["a"],
            "can_buy": [False],
            "can_sell": [True],
            "reason": ["blocked"],
        }
    )
    transaction = plans([("b1", D1, "a", 5)])
    result = evaluate_execution(
        transaction, prices, initial_capital=100, execution_availability=availability
    )
    assert result.tables["fills"].is_empty()
    assert result.tables["orders"]["status"].to_list() == ["expired"]
    assert result.tables["transaction_pnl"]["actual_return_reason"].to_list() == [
        "no_filled_notional"
    ]
    retry = evaluate_execution(
        transaction,
        prices,
        initial_capital=100,
        execution_availability=availability,
        config=ExecutionConfig(retry_unfilled=True),
    )
    assert retry.tables["fills"]["time"].to_list() == [D2]
    assert retry.tables["fills"]["quantity"].to_list() == [5]
    assert retry.tables["ideal_fills"]["time"].to_list() == [D1]
    assert retry.tables["fills"]["commission"].to_list() == [5]


def test_supplied_lots_settlement_and_neutral_defaults():
    transaction = plans([("b1", D1, "a", 5), ("s1", D1, "a", -1)])
    # Sell-first makes this reference structurally infeasible, independent
    # of the actual same-day settlement setting.
    with pytest.raises(InputValidationError, match="short inventory"):
        evaluate_execution(
            transaction, market([(D1, "a", 10, 10)]), initial_capital=100
        )
    result = evaluate_execution(
        plans([("b1", D1, "a", 5), ("s1", D2, "a", -5)]),
        market([(D1, "a", 10, 10), (D2, "a", 10, 10), (D3, "a", 10, 10)]),
        initial_capital=100,
        config=ExecutionConfig(settlement_sessions=2),
        lot_sizes=pl.DataFrame({"asset_id": ["a"], "buy_lot_size": [2]}),
    )
    assert result.tables["fills"]["quantity"].to_list() == [4]
    assert result.tables["fills"]["stamp_tax"].to_list() == [0]
    assert result.tables["fills"]["slippage_cost"].to_list() == [0]


def test_custom_identified_cost_rule_and_quote_validation():
    rule = ExecutionCostRule(
        "flat",
        "1",
        {"money_fee": 2},
        lambda _side, _qty, price, _day, _asset: ExecutionCostQuote(2, price),
    )
    result = evaluate_execution(
        plans([("b1", D1, "a", 5)]),
        market([(D1, "a", 10, 10)]),
        initial_capital=100,
        config=ExecutionConfig(cost_rule=rule),
    )
    assert result.tables["fills"]["explicit_cost"].to_list() == [2]
    assert result.checkpoint.actual.cash == 48
    with pytest.raises(InputValidationError, match="stable ID"):
        ExecutionCostRule("", "1", {}, rule.quote)
    bad = ExecutionCostRule(
        "bad",
        "1",
        {},
        lambda _side, _qty, price, _day, _asset: ExecutionCostQuote(-1, price),
    )
    with pytest.raises(InputValidationError, match="cost quote"):
        evaluate_execution(
            plans([("b1", D1, "a", 5)]),
            market([(D1, "a", 10, 10)]),
            initial_capital=100,
            config=ExecutionConfig(cost_rule=bad),
        )


def test_mark_gaps_freeze_observed_prices_and_do_not_read_future():
    result = evaluate_execution(
        plans([("b1", D1, "a", 5)]),
        market([(D1, "a", 10, 10), (D2, "a", None, None), (D3, "a", 30, 30)]),
        initial_capital=100,
        available_date=D2,
    )
    assert result.tables["account_value"]["equity"].to_list() == [95, 95]
    assert result.tables["positions"]["mark_price"].to_list() == [10, 10]


def test_continuation_codec_and_prefix_verification_match_full_run():
    transaction = plans([("b1", D1, "a", 5), ("s1", D3, "a", -3)])
    prices = market([(D1, "a", 10, 10), (D2, "a", 12, 12), (D3, "a", 15, 15)])
    first = evaluate_execution(
        transaction, prices, initial_capital=100, available_date=D2
    )
    frames, metadata = save_checkpoint(first.checkpoint)
    restored = load_checkpoint(frames, metadata)
    continued = evaluate_execution(
        transaction, prices, initial_capital=100, checkpoint=restored
    )
    full = evaluate_execution(transaction, prices, initial_capital=100)
    assert_frame_equal(
        continued.tables["account_value"],
        full.tables["account_value"].filter(pl.col("time") > D2),
    )
    assert_frame_equal(continued.checkpoint.actual_lots, full.checkpoint.actual_lots)
    changed = prices.with_columns(
        pl.when(pl.col("time") == D1)
        .then(11)
        .otherwise(pl.col("execution_price"))
        .alias("execution_price")
    )
    with pytest.raises(InputValidationError, match="causal prefix changed"):
        evaluate_execution(
            transaction, changed, initial_capital=100, checkpoint=restored
        )


def test_causal_weight_bridge_freezes_actual_state_deltas_before_open_gap():
    weights = pl.DataFrame({"time": [D1], "asset_id": ["a"], "value": [1.0]})
    decisions = pl.DataFrame({"time": [D1, D2], "status": ["rebalance", "hold"]})
    calendar = pl.DataFrame({"time": [D1, D2, D3]})
    # Plan is 10 shares at decision close 10. The next opening price 9
    # must not change that frozen quantity to 11 shares.
    result = run_execution_from_weights(
        weights,
        decisions,
        market([(D1, "a", 10, 10), (D2, "a", 9, 9), (D3, "a", 9, 9)]),
        initial_capital=100,
        calendar=calendar,
    )
    assert result.tables["transaction_plans"]["quantity"].to_list() == [10]
    assert result.tables["fills"]["quantity"].to_list() == [10]
    assert result.tables["fills"]["time"].to_list() == [D2]


def test_weight_bridge_future_plan_checkpoint_never_replays_decision():
    weights = pl.DataFrame({"time": [D1], "asset_id": ["a"], "value": [1.0]})
    decisions = pl.DataFrame({"time": [D1], "status": ["rebalance"]})
    prices = market([(D1, "a", 10, 10), (D2, "a", 9, 9), (D3, "a", 9, 9)])
    calendar = pl.DataFrame({"time": [D1, D2, D3]})
    first = run_execution_from_weights(
        weights,
        decisions,
        prices,
        initial_capital=100,
        calendar=calendar,
        available_date=D1,
    )
    assert first.checkpoint.pending_transactions["quantity"].to_list() == [10]
    continued = run_execution_from_weights(
        weights,
        decisions,
        prices,
        initial_capital=100,
        calendar=calendar,
        checkpoint=first.checkpoint,
    )
    assert continued.tables["transaction_plans"]["quantity"].to_list() == [10]
    assert continued.tables["fills"].height == 1


def test_dividends_receivables_and_bonus_shares_reconcile_fifo():
    actions = pl.DataFrame(
        {
            "action_id": ["div"],
            "asset_id": ["a"],
            "is_implemented": [True],
            "record_date": [D1],
            "ex_date": [D2],
            "cash_pay_date": [D3],
            "share_available_date": [D3],
            "cash_dividend_per_share": [1.0],
            "stock_dividend_per_share": [0.2],
        }
    )
    result = evaluate_execution(
        plans([("b1", D1, "a", 5), ("s1", D4, "a", -6)]),
        market([(D1, "a", 10, 10), (D2, "a", 8, 8), (D3, "a", 8, 8), (D4, "a", 8, 8)]),
        initial_capital=100,
        corporate_actions=actions,
    )
    assert result.tables["account_value"]["equity"].to_list() == [95, 98, 98, 93]
    assert result.tables["account_value"]["dividend_income"][-1] == 5
    assert result.tables["account_value"]["reconciliation_error"].abs().max() < 1e-9
    assert result.tables["open_lots"].is_empty()


def test_stable_order_and_logical_hash_are_chunk_and_input_order_independent():
    transaction = plans([("b2", D1, "b", 4), ("b1", D1, "a", 4)])
    prices = market([(D1, "b", 10, 11), (D1, "a", 10, 12)])
    first = evaluate_execution(transaction, prices, initial_capital=100)
    second = evaluate_execution(
        transaction.reverse(), prices.reverse(), initial_capital=100
    )
    assert_frame_equal(first.tables["fills"], second.tables["fills"])
    assert logical_content_hash(prices) == logical_content_hash(prices.reverse())
    assert logical_content_hash(
        pl.concat([prices.head(1), prices.tail(1)], rechunk=False)
    ) == logical_content_hash(prices)


def test_duplicate_plan_id_is_rejected_without_hidden_netting():
    with pytest.raises(InputValidationError, match="plan_id must be unique"):
        evaluate_execution(
            plans([("same", D1, "a", 4), ("same", D2, "a", -4)]),
            market([(D1, "a", 10, 10), (D2, "a", 10, 10)]),
            initial_capital=100,
        )


def test_full_and_checkpoint_append_have_identical_all_public_tables():
    transaction = plans(
        [
            ("b1", D1, "a", 4),
            ("b2", D2, "a", 3),
            ("s1", D3, "a", -5),
        ]
    )
    prices = market([(D1, "a", 10, 10), (D2, "a", 20, 20), (D3, "a", 30, 30)])
    first = evaluate_execution(
        transaction, prices, initial_capital=200, available_date=D2
    )
    later = evaluate_execution(
        transaction, prices, initial_capital=200, checkpoint=first.checkpoint
    )
    complete = evaluate_execution(transaction, prices, initial_capital=200)
    merged = merge_execution_tables(first.tables, later.tables)
    for name, table in complete.tables.items():
        assert_frame_equal(merged[name], table)
    with pytest.raises(InputValidationError, match="overlaps"):
        merge_execution_tables(first.tables, complete.tables)


def test_retry_unrealized_summary_adds_all_fills_for_one_original_plan():
    result = evaluate_execution(
        plans([("b1", D1, "a", 10)]),
        market([(D1, "a", 10, 10), (D2, "a", 5, 5)]),
        initial_capital=100,
        config=ExecutionConfig(min_fee=1, retry_unfilled=True),
    )
    assert result.tables["fills"]["quantity"].to_list() == [9, 1]
    row = result.tables["transaction_pnl"].row(0, named=True)
    assert row["actual_quantity"] == 10
    assert row["actual_cost"] == 2
    assert row["actual_pnl"] == -47
    assert row["target_pnl"] == -50


def test_cash_only_no_trade_outputs_have_public_typed_schemas():
    result = evaluate_execution(
        plans([]), market([(D1, "a", 10, 10)]), initial_capital=100
    )
    assert result.tables["fills"].schema["quantity"] == pl.Int64
    assert result.tables["orders"].schema["expires_at"] == pl.Date
    assert result.tables["fifo_matches"].schema["entry_price"] == pl.Float64
    assert result.tables["positions"].schema["close_price"] == pl.Float64
    assert result.tables["account_value"]["equity"].to_list() == [100]


def test_frozen_future_execution_calendar_and_declared_delta_cannot_change():
    weights = pl.DataFrame({"time": [D1], "asset_id": ["a"], "value": [1.0]})
    decisions = pl.DataFrame({"time": [D1], "status": ["rebalance"]})
    prices = market([(D1, "a", 10, 10), (D2, "a", 9, 9), (D3, "a", 9, 9)])
    calendar = pl.DataFrame({"time": [D1, D2, D3]})
    first = run_execution_from_weights(
        weights,
        decisions,
        prices,
        initial_capital=100,
        calendar=calendar,
        available_date=D1,
    )
    with pytest.raises(InputValidationError, match="causal prefix changed"):
        run_execution_from_weights(
            weights,
            decisions,
            prices.filter(pl.col("time") != D2),
            initial_capital=100,
            calendar=calendar.filter(pl.col("time") != D2),
            checkpoint=first.checkpoint,
        )
    explicit = plans([("b1", D2, "a", 5)]).with_columns(
        pl.lit(D1).alias("decision_date")
    )
    saved = evaluate_execution(explicit, prices, initial_capital=100, available_date=D1)
    changed = explicit.with_columns(pl.lit(6).alias("quantity"))
    with pytest.raises(InputValidationError, match="causal prefix changed"):
        evaluate_execution(
            changed, prices, initial_capital=100, checkpoint=saved.checkpoint
        )


def test_newly_mature_record_date_entitlement_needs_both_verified_histories():
    transaction = plans([("b1", D1, "a", 5)])
    prices = market([(D1, "a", 10, 10), (D2, "a", 9, 9), (D3, "a", 9, 9)])
    actions = pl.DataFrame(
        {
            "action_id": ["div"],
            "asset_id": ["a"],
            "is_implemented": [True],
            "record_date": [D1],
            "ex_date": [D2],
            "cash_pay_date": [D3],
            "share_available_date": [None],
            "cash_dividend_per_share": [1.0],
            "stock_dividend_per_share": [0.0],
        }
    )
    first = evaluate_execution(
        transaction, prices, initial_capital=100, available_date=D1
    )
    with pytest.raises(InputValidationError, match="record-date position history"):
        evaluate_execution(
            transaction,
            prices,
            initial_capital=100,
            checkpoint=first.checkpoint,
            corporate_actions=actions,
        )
    later = evaluate_execution(
        transaction,
        prices,
        initial_capital=100,
        checkpoint=first.checkpoint,
        corporate_actions=actions,
        historical_positions=first.tables["positions"],
        historical_ideal_positions=first.tables["ideal_positions"],
        historical_sessions=first.tables["account_value"].select("time"),
        historical_open_lots=first.tables["open_lot_history"],
        historical_ideal_open_lots=first.tables["ideal_open_lot_history"],
    )
    complete = evaluate_execution(
        transaction, prices, initial_capital=100, corporate_actions=actions
    )
    assert_frame_equal(
        later.tables["account_value"],
        complete.tables["account_value"].filter(pl.col("time") > D1),
    )


def test_stress_scenarios_scale_only_declared_component_and_not_minimum_fee():
    config = ExecutionConfig(capital_mode="fixed_notional", fixed_notional=100)
    scenarios = execution_stress_scenarios(100, config, session_lag=1)
    assert len(scenarios) == 9
    by_name = {
        name: (capital, settings, lag) for name, capital, settings, lag in scenarios
    }
    assert by_name["capital_5"][0] == 500
    assert by_name["capital_5"][1].fixed_notional == 500
    assert by_name["commission_4"][1].rate == 0.002
    assert by_name["commission_4"][1].min_fee == 5
    assert by_name["slippage_4"][1].buy_slippage_rate == 0
    assert by_name["delay_4"][2] == 5


def test_historical_transaction_summary_uses_saved_cutoff_and_empty_final_inventory():
    transaction = plans([("b1", D1, "a", 5), ("s1", D3, "a", -5)])
    prices = market([(D1, "a", 10, 10), (D2, "a", 20, 20), (D3, "a", 30, 30)])
    complete = evaluate_execution(transaction, prices, initial_capital=100)
    first = evaluate_execution(
        transaction, prices, initial_capital=100, available_date=D1
    )
    assert_frame_equal(
        summarize_transaction_pnl(complete.tables, through=D1),
        first.tables["transaction_pnl"],
    )
    assert summarize_transaction_pnl(complete.tables, through=D1)["actual_pnl"][0] == -5
    assert_frame_equal(
        summarize_transaction_pnl(complete.tables, through=D3),
        complete.tables["transaction_pnl"],
    )
    assert complete.tables["open_lot_history"].filter(pl.col("time") == D3).is_empty()


def test_per_transaction_dividend_and_pending_bonus_pnl_reconcile_at_cutoff():
    actions = pl.DataFrame(
        {
            "action_id": ["div"],
            "asset_id": ["a"],
            "is_implemented": [True],
            "record_date": [D1],
            "ex_date": [D2],
            "cash_pay_date": [D3],
            "share_available_date": [D3],
            "cash_dividend_per_share": [1.0],
            "stock_dividend_per_share": [0.2],
        }
    )
    result = evaluate_execution(
        plans([("b1", D1, "a", 5), ("s1", D4, "a", -6)]),
        market([(D1, "a", 10, 10), (D2, "a", 8, 8), (D3, "a", 8, 8), (D4, "a", 8, 8)]),
        initial_capital=100,
        corporate_actions=actions,
    )
    at_ex = summarize_transaction_pnl(result.tables, through=D2)
    assert at_ex["actual_dividend_income"].sum() == 5
    assert at_ex["actual_pnl"].sum() == -2
    assert at_ex["target_pnl"].sum() == 3
    assert result.tables["transaction_pnl"]["actual_pnl"].sum() == -7
    assert result.tables["corporate_pnl"]["plan_id"].to_list() == ["b1", "b1"]


@pytest.mark.parametrize("blocked", [False, True])
def test_retry_survives_codec_and_every_result_table_matches_full_run(blocked):
    prices = market([(D1, "a", 10, 10), (D2, "a", 5, 5)])
    transaction = plans([("b1", D1, "a", 10)])
    availability = (
        pl.DataFrame(
            {
                "time": [D1],
                "asset_id": ["a"],
                "can_buy": [False],
                "can_sell": [True],
                "reason": ["blocked"],
            }
        )
        if blocked
        else None
    )
    settings = dict(
        initial_capital=100,
        execution_availability=availability,
        config=ExecutionConfig(min_fee=1, retry_unfilled=True),
    )
    first = evaluate_execution(transaction, prices, available_date=D1, **settings)
    assert first.checkpoint.pending_transactions["_retry"].to_list() == [True]
    frames, metadata = save_checkpoint(first.checkpoint)
    later = evaluate_execution(
        transaction, prices, checkpoint=load_checkpoint(frames, metadata), **settings
    )
    complete = evaluate_execution(transaction, prices, **settings)
    merged = merge_execution_tables(first.tables, later.tables)
    for name, table in complete.tables.items():
        assert_frame_equal(merged[name], table)


def test_cooperative_cancellation_stops_before_later_sessions_without_result():
    seen = []

    def cancel():
        if seen:
            raise RuntimeError("canceled")

    def progress(completed, _total):
        seen.append(completed)

    with pytest.raises(RuntimeError, match="canceled"):
        evaluate_execution(
            plans([("b1", D1, "a", 5)]),
            market([(D1, "a", 10, 10), (D2, "a", 11, 11)]),
            initial_capital=100,
            check_canceled=cancel,
            progress=progress,
        )
    assert seen == [1]


def test_settlement_pending_sale_retries_only_residual_after_availability():
    result = evaluate_execution(
        plans([("b1", D1, "a", 5), ("s1", D2, "a", -5)]),
        market([(D1, "a", 10, 10), (D2, "a", 10, 10), (D3, "a", 10, 10)]),
        initial_capital=100,
        config=ExecutionConfig(settlement_sessions=2, retry_unfilled=True),
    )
    assert result.tables["fills"]["time"].to_list() == [D1, D3]
    assert result.tables["fills"]["quantity"].to_list() == [5, 5]
    assert (
        result.tables["orders"]["reason"].to_list()[1]
        == "insufficient_available_inventory"
    )


def test_distinct_slippage_and_sell_tax_are_explicit_and_reconcile():
    result = evaluate_execution(
        plans([("b1", D1, "a", 5), ("s1", D2, "a", -5)]),
        market([(D1, "a", 10, 10), (D2, "a", 10, 10)]),
        initial_capital=100,
        config=ExecutionConfig(
            rate=0,
            min_fee=0,
            buy_slippage_rate=0.01,
            sell_slippage_rate=0.02,
            stamp_tax_rate=0.001,
        ),
    )
    assert result.tables["fills"]["fill_price"].to_list() == pytest.approx([10.1, 9.8])
    assert result.tables["fills"]["stamp_tax"].to_list() == pytest.approx([0, 0.049])
    assert result.metrics["actual_pnl"] == pytest.approx(-1.549)


def test_corporate_action_coverage_is_explicit_and_complete_when_supplied():
    with pytest.raises(InputValidationError, match="coverage is incomplete"):
        evaluate_execution(
            plans([("b1", D1, "a", 5)]),
            market([(D1, "a", 10, 10), (D2, "a", 10, 10)]),
            initial_capital=100,
            corporate_action_coverage=pl.DataFrame(
                {"time": [D1], "is_complete": [True]}
            ),
        )


def test_checkpoint_retains_exited_asset_coordinates_for_causal_actions():
    result = evaluate_execution(
        plans([("b1", D1, "a", 5), ("s1", D2, "a", -5)]),
        market([(D1, "a", 10, 10), (D2, "a", 10, 10)]),
        initial_capital=100,
    )
    assert result.checkpoint.actual.positions["quantity"].to_list() == [0]
    assert result.checkpoint.ideal.positions["quantity"].to_list() == [0]
    assert result.tables["positions"].filter(pl.col("time") == D2).is_empty()


@pytest.mark.parametrize(
    "column,value",
    [
        ("cash_dividend_per_share", -1.0),
        ("stock_dividend_per_share", float("nan")),
        ("cash_dividend_per_share", float("inf")),
        ("record_date", D2),
        ("cash_pay_date", D1),
        ("share_available_date", D1),
    ],
)
def test_invalid_corporate_amounts_or_chronology_fail_at_boundary(column, value):
    actions = pl.DataFrame(
        {
            "action_id": ["div"],
            "asset_id": ["a"],
            "is_implemented": [True],
            "record_date": [D1],
            "ex_date": [D2],
            "cash_pay_date": [D3],
            "share_available_date": [D3],
            "cash_dividend_per_share": [1.0],
            "stock_dividend_per_share": [0.2],
        }
    ).with_columns(pl.lit(value).alias(column))
    with pytest.raises(
        InputValidationError, match="finite nonnegative amounts and causal dates"
    ):
        evaluate_execution(
            plans([("b1", D1, "a", 5)]),
            market([(D1, "a", 10, 10), (D2, "a", 8, 8), (D3, "a", 8, 8)]),
            initial_capital=100,
            corporate_actions=actions,
        )


def test_zero_never_held_coordinate_needs_no_decision_price():
    weights = pl.DataFrame(
        {"time": [D1, D1], "asset_id": ["a", "never_held"], "value": [0.5, 0.0]}
    )
    result = run_execution_from_weights(
        weights,
        pl.DataFrame({"time": [D1], "status": ["rebalance"]}),
        market([(D1, "a", 10, 10), (D2, "a", 10, 10)]),
        initial_capital=100,
        calendar=pl.DataFrame({"time": [D1, D2]}),
    )
    assert result.tables["transaction_plans"]["asset_id"].to_list() == ["a"]
    assert result.tables["transaction_plans"]["quantity"].to_list() == [5]


def test_explicit_reference_price_differs_from_actual_execution_without_resize():
    transaction = plans([("b1", D1, "a", 10)]).with_columns(
        pl.lit(10.0).alias("reference_price")
    )
    prices = market([(D1, "a", 12, 12)]).with_columns(
        pl.lit(9.0).alias("reference_price")
    )
    result = evaluate_execution(transaction, prices, initial_capital=100)
    assert result.tables["ideal_fills"]["quantity"].to_list() == [10]
    assert result.tables["ideal_fills"]["fill_price"].to_list() == [10]
    assert result.tables["fills"]["quantity"].to_list() == [7]
    assert result.tables["fills"]["fill_price"].to_list() == [12]
    assert result.metrics["target_pnl"] == 20
    assert result.metrics["actual_pnl"] == -5


def test_weight_bridge_explicit_decision_reference_survives_gap_and_checkpoint():
    weights = pl.DataFrame({"time": [D1], "asset_id": ["a"], "value": [1.0]})
    decisions = pl.DataFrame({"time": [D1], "status": ["rebalance"]})
    prices = market([(D1, "a", 10, 10), (D2, "a", 20, 20)])
    options = dict(
        initial_capital=100,
        calendar=pl.DataFrame({"time": [D1, D2]}),
        reference_price_mode="decision",
    )
    first = run_execution_from_weights(
        weights, decisions, prices, available_date=D1, **options
    )
    later = run_execution_from_weights(
        weights, decisions, prices, checkpoint=first.checkpoint, **options
    )
    complete = run_execution_from_weights(weights, decisions, prices, **options)
    assert complete.tables["transaction_plans"]["quantity"].to_list() == [10]
    assert complete.tables["transaction_plans"]["reference_price"].to_list() == [10]
    assert complete.tables["fills"]["quantity"].to_list() == [4]
    assert complete.tables["ideal_fills"]["fill_price"].to_list() == [10]
    for name, frame in complete.tables.items():
        assert_frame_equal(
            merge_execution_tables(first.tables, later.tables)[name], frame
        )
    with pytest.raises(InputValidationError, match="cash-infeasible"):
        run_execution_from_weights(
            weights,
            decisions,
            prices,
            initial_capital=100,
            calendar=options["calendar"],
            reference_price_mode="execution",
        )
