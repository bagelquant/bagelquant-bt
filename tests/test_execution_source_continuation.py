"""Native source authority admits only new account sessions and immutable tails."""

import hashlib
from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt import (
    BTStore,
    ExecutionConfig,
    InputValidationError,
    SharedInputEvidence,
    evaluate_execution,
    load_checkpoint,
    merge_execution_tables,
    run_execution_from_weights,
    save_checkpoint,
    validate_execution_continuation,
)
from test_execution_evaluation import market, plans

START = date(2024, 1, 2)
DAYS = [START + timedelta(days=index) for index in range(8)]


def evidence(through, *, mode="weights", receipt="original"):
    channels = {
        "prices",
        "calendar",
        "availability",
        "lot_sizes",
        "actions",
        "action_coverage",
        *({"weights", "decisions"} if mode == "weights" else {"transactions"}),
    }
    return {
        name: SharedInputEvidence(
            source_receipt=f"{receipt}:{name}",
            source_digest=hashlib.sha256(f"{receipt}:{name}".encode()).hexdigest(),
            content_token=hashlib.sha256(f"{name}:{through}".encode()).hexdigest(),
            start=START,
            end=DAYS[-1] if name == "calendar" else through,
            source_information_cutoff=through,
            schema={"time": "Date", "value": "Float64"},
            selector={"view": "consumed", "strict": True, "empty_coordinates": True},
            complete=True,
        )
        for name in channels
    }


def bridge_inputs():
    weights = pl.DataFrame(
        {"time": [DAYS[0], DAYS[3]], "asset_id": ["a", "a"], "value": [1.0, 0.0]}
    )
    decisions = pl.DataFrame(
        {
            "time": DAYS[:6],
            "status": ["rebalance", "hold", "hold", "rebalance", "hold", "hold"],
        }
    )
    prices = market(
        [(day, "a", 10.0 + index, 10.0 + index) for index, day in enumerate(DAYS[:6])]
    )
    calendar = pl.DataFrame({"time": DAYS})
    return weights, decisions, prices, calendar


def test_native_continuation_rejects_changed_static_rule_on_consumed_asset():
    weights, decisions, prices, calendar = bridge_inputs()
    settings = dict(
        initial_capital=100.0,
        calendar=calendar,
        reference_price_mode="decision",
        config=ExecutionConfig(rate=0, min_fee=0),
    )
    first = run_execution_from_weights(
        weights.head(1),
        decisions.head(2),
        prices.head(2),
        available_date=DAYS[1],
        input_evidence=evidence(DAYS[1]),
        lot_sizes=pl.DataFrame({"asset_id": ["a"], "buy_lot_size": [1]}),
        **settings,
    )
    with pytest.raises(InputValidationError, match="consumed asset's lot rule"):
        run_execution_from_weights(
            weights.filter(pl.col("time") > DAYS[1]),
            decisions.slice(2, 1),
            prices.slice(2, 1),
            checkpoint=first.checkpoint,
            available_date=DAYS[2],
            input_evidence=evidence(DAYS[2]),
            prefix_evidence=evidence(DAYS[1]),
            lot_sizes=pl.DataFrame({"asset_id": ["a"], "buy_lot_size": [2]}),
            **settings,
        )


def test_native_bridge_tail_matches_full_and_retains_cumulative_plan_pnl(monkeypatch):
    weights, decisions, prices, calendar = bridge_inputs()
    settings = dict(
        initial_capital=100.0,
        calendar=calendar,
        reference_price_mode="decision",
        config=ExecutionConfig(rate=0, min_fee=0, settlement_sessions=1),
    )
    first = run_execution_from_weights(
        weights.filter(pl.col("time") <= DAYS[2]),
        decisions.head(3),
        prices.head(3),
        available_date=DAYS[2],
        input_evidence=evidence(DAYS[2]),
        **settings,
    )
    encoded, metadata = save_checkpoint(first.checkpoint)
    restored = load_checkpoint(encoded, metadata)
    proof = evidence(DAYS[2], receipt="current")
    # Admission uses original scope and source tokens, not original receipt IDs
    # or numerical frames; annualization alone remains financially neutral.
    validate_execution_continuation(
        metadata,
        prefix_evidence=proof,
        initial_capital=100.0,
        config=replace(settings["config"], annualization=240),
        reference_price_mode="decision",
    )
    sessions = []
    later = run_execution_from_weights(
        weights.filter(pl.col("time") > DAYS[2]),
        decisions.tail(3),
        prices.tail(3),
        checkpoint=restored,
        available_date=DAYS[5],
        input_evidence=evidence(DAYS[5], receipt="current"),
        prefix_evidence=proof,
        progress=lambda completed, total: sessions.append((completed, total)),
        **settings,
    )
    full = run_execution_from_weights(
        weights, decisions, prices, available_date=DAYS[5], **settings
    )
    assert sessions == [(1, 3), (2, 3), (3, 3)]
    merged = merge_execution_tables(first.tables, later.tables)
    for name, expected in full.tables.items():
        assert_frame_equal(merged[name], expected)
    assert_frame_equal(later.tables["transaction_pnl"], full.tables["transaction_pnl"])
    assert_frame_equal(
        later.checkpoint.transaction_summary, full.checkpoint.transaction_summary
    )
    assert restored.actual.positions["last_mark"].to_list() == [12.0]


@pytest.mark.parametrize(
    "failure", ["missing", "changed", "incomplete", "settings", "damaged"]
)
def test_unknown_or_changed_prefix_blocks_before_numerical_input_validation(
    monkeypatch, failure
):
    weights, decisions, prices, calendar = bridge_inputs()
    config = ExecutionConfig(rate=0, min_fee=0)
    first = run_execution_from_weights(
        weights.head(1),
        decisions.head(1),
        prices.head(1),
        initial_capital=100,
        calendar=calendar,
        available_date=DAYS[0],
        config=config,
        input_evidence=evidence(DAYS[0]),
    )
    prefix = evidence(DAYS[0], receipt="current")
    checkpoint = first.checkpoint
    if failure == "missing":
        checkpoint = replace(checkpoint, source_binding=None)
    elif failure == "changed":
        prefix["prices"] = replace(prefix["prices"], content_token="0" * 64)
    elif failure == "incomplete":
        prefix["prices"] = replace(prefix["prices"], complete=False)
    elif failure == "settings":
        config = replace(config, min_fee=5)
    elif failure == "damaged":
        checkpoint = replace(checkpoint, prefix_hash="0" * 64)
    from bagelquant_bt import execution_evaluation

    monkeypatch.setattr(
        execution_evaluation,
        "_market",
        lambda *_: pytest.fail("unproven history reached value validation"),
    )
    with pytest.raises(
        InputValidationError, match=r"(maintenance|prefix|incomplete|settings|digest)"
    ):
        run_execution_from_weights(
            weights.tail(1),
            decisions.tail(1),
            prices.tail(1),
            initial_capital=100,
            calendar=calendar,
            checkpoint=checkpoint,
            config=config,
            input_evidence=evidence(DAYS[5]),
            prefix_evidence=prefix,
        )


def test_native_explicit_plans_restore_pending_and_only_read_tail():
    config = ExecutionConfig(
        rate=0, min_fee=0, retry_unfilled=True, settlement_sessions=2
    )
    calendar = pl.DataFrame({"time": DAYS})
    transactions = plans(
        [("buy", DAYS[1], "a", 5), ("sell", DAYS[4], "a", -5)]
    ).with_columns(pl.Series("decision_date", [DAYS[0], DAYS[3]]))
    prices = market(
        [(day, "a", 10.0, 10.0 + index) for index, day in enumerate(DAYS[:6])]
    )
    availability = pl.DataFrame(
        {
            "time": [DAYS[1]],
            "asset_id": ["a"],
            "can_buy": [False],
            "can_sell": [True],
            "reason": ["blocked"],
        }
    )
    first = evaluate_execution(
        transactions.head(1),
        prices.head(1),
        initial_capital=100,
        calendar=calendar,
        config=config,
        available_date=DAYS[0],
        input_evidence=evidence(DAYS[0], mode="transactions"),
    )
    assert first.checkpoint.pending_transactions["plan_id"].to_list() == ["buy"]
    continued = evaluate_execution(
        transactions.tail(1),
        prices.tail(5),
        initial_capital=100,
        calendar=calendar,
        config=config,
        available_date=DAYS[5],
        checkpoint=first.checkpoint,
        execution_availability=availability,
        input_evidence=evidence(DAYS[5], mode="transactions"),
        prefix_evidence=evidence(DAYS[0], mode="transactions", receipt="new"),
    )
    full = evaluate_execution(
        transactions,
        prices,
        initial_capital=100,
        calendar=calendar,
        config=config,
        execution_availability=availability,
        available_date=DAYS[5],
    )
    for name, frame in merge_execution_tables(first.tables, continued.tables).items():
        assert_frame_equal(frame, full.tables[name])
    assert_frame_equal(
        continued.tables["transaction_pnl"], full.tables["transaction_pnl"]
    )


def test_execution_reference_append_and_bounded_concat_read_have_zero_old_io(
    tmp_path, monkeypatch
):
    weights, decisions, prices, calendar = bridge_inputs()
    options = dict(
        initial_capital=100,
        calendar=calendar,
        reference_price_mode="decision",
        config=ExecutionConfig(rate=0, min_fee=0),
    )
    first = run_execution_from_weights(
        weights.head(1),
        decisions.head(3),
        prices.head(3),
        available_date=DAYS[2],
        input_evidence=evidence(DAYS[2]),
        **options,
    )
    later = run_execution_from_weights(
        weights.tail(1),
        decisions.tail(3),
        prices.tail(3),
        available_date=DAYS[5],
        checkpoint=first.checkpoint,
        input_evidence=evidence(DAYS[5]),
        prefix_evidence=evidence(DAYS[2]),
        **options,
    )
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frames, state = save_checkpoint(first.checkpoint)
    old = store.publish_shared(
        "weights",
        "account",
        "old",
        {**first.tables, **frames},
        metadata={"checkpoint": state},
    )
    original = {
        name: store._table_path(ref["table_id"]).read_bytes()
        for name, ref in old["files"].items()
    }
    later_frames, later_state = save_checkpoint(later.checkpoint)
    with monkeypatch.context() as patch:
        patch.setattr(
            pl,
            "read_parquet",
            lambda *_a, **_k: pytest.fail("append decoded old table"),
        )
        refs = store.execution_references(
            {**later.tables, **later_frames}, previous=old["files"], through=DAYS[2]
        )
        current = store.publish_shared(
            "weights",
            "account",
            "new",
            {},
            file_references=refs,
            metadata={"checkpoint": later_state},
            expected_shared=old,
        )
    assert all(
        store._table_path(old["files"][name]["table_id"]).read_bytes() == raw
        for name, raw in original.items()
    )
    reads = []
    scan = pl.scan_parquet
    monkeypatch.setattr(
        pl,
        "scan_parquet",
        lambda path, **kwargs: (reads.append(str(path)), scan(path, **kwargs))[1],
    )
    selected = store.read_reference(
        current["files"]["positions"],
        filters=[{"column": "time", "operation": "eq", "value": DAYS[3]}],
    )
    assert_frame_equal(
        selected,
        later.tables["positions"].filter(pl.col("time") == DAYS[3]),
        check_column_order=False,
    )
    assert len(reads) == 1 and old["files"]["positions"]["table_id"] not in reads[0]
    reads.clear()
    assert store.read_reference(
        current["files"]["positions"],
        filters=[{"column": "time", "operation": "eq", "value": DAYS[5]}],
    ).is_empty()
    assert not reads
    with pytest.raises(ValueError, match="overlaps"):
        store.execution_references(
            {"returns": first.tables["returns"]},
            previous=current["files"],
            through=DAYS[5],
        )
    assert store.verify()["status"] == "complete"


def test_new_action_uses_exact_record_history_and_saved_future_entitlements():
    calendar = pl.DataFrame({"time": DAYS})
    prices = market([(day, "a", 10.0, 10.0) for day in DAYS[:4]])
    transaction = plans([("buy", DAYS[0], "a", 5)])
    config = ExecutionConfig(rate=0, min_fee=0)
    first = evaluate_execution(
        transaction,
        prices.head(1),
        initial_capital=100,
        calendar=calendar,
        config=config,
        available_date=DAYS[0],
        input_evidence=evidence(DAYS[0], mode="transactions"),
    )
    actions = pl.DataFrame(
        {
            "action_id": ["distribution"],
            "asset_id": ["a"],
            "is_implemented": [True],
            "record_date": [DAYS[0]],
            "ex_date": [DAYS[1]],
            "cash_pay_date": [DAYS[2]],
            "share_available_date": [DAYS[3]],
            "cash_dividend_per_share": [1.0],
            "stock_dividend_per_share": [0.2],
        }
    )
    empty_plans = transaction.head(0)
    kwargs = dict(
        initial_capital=100,
        calendar=calendar,
        config=config,
        checkpoint=first.checkpoint,
        corporate_actions=actions,
        available_date=DAYS[1],
        input_evidence=evidence(DAYS[1], mode="transactions"),
        prefix_evidence=evidence(DAYS[0], mode="transactions"),
    )
    with pytest.raises(InputValidationError, match="record-date"):
        evaluate_execution(empty_plans, prices.slice(1, 1), **kwargs)
    histories = {
        "historical_positions": first.tables["positions"],
        "historical_ideal_positions": first.tables["ideal_positions"],
        "historical_sessions": first.tables["account_value"].select("time"),
        "historical_open_lots": first.tables["open_lot_history"],
        "historical_ideal_open_lots": first.tables["ideal_open_lot_history"],
    }
    second = evaluate_execution(empty_plans, prices.slice(1, 1), **histories, **kwargs)
    assert second.checkpoint.active_actions.height == 1
    third = evaluate_execution(
        empty_plans,
        prices.tail(2),
        initial_capital=100,
        calendar=calendar,
        config=config,
        checkpoint=second.checkpoint,
        available_date=DAYS[3],
        input_evidence=evidence(DAYS[3], mode="transactions"),
        prefix_evidence=evidence(DAYS[1], mode="transactions"),
    )
    full = evaluate_execution(
        transaction,
        prices,
        initial_capital=100,
        calendar=calendar,
        config=config,
        corporate_actions=actions,
    )
    merged = merge_execution_tables(
        merge_execution_tables(first.tables, second.tables), third.tables
    )
    for name, expected in full.tables.items():
        assert_frame_equal(merged[name], expected)
    assert_frame_equal(third.tables["transaction_pnl"], full.tables["transaction_pnl"])
    assert third.checkpoint.active_actions.is_empty()


def test_native_suffix_repair_prunes_old_daily_refs_and_keeps_receipts(tmp_path):
    calendar = pl.DataFrame({"time": DAYS})
    transaction = plans([("buy", DAYS[0], "a", 5), ("sell", DAYS[4], "a", -5)])
    prices = market(
        [(day, "a", 10.0, 10.0 + index) for index, day in enumerate(DAYS[:6])]
    )
    config = ExecutionConfig(rate=0, min_fee=0)
    args = dict(initial_capital=100, calendar=calendar, config=config)
    prefix = evaluate_execution(
        transaction.head(1),
        prices.head(2),
        available_date=DAYS[1],
        input_evidence=evidence(DAYS[1], mode="transactions"),
        **args,
    )
    old = evaluate_execution(transaction, prices, **args)
    changed = prices.with_columns(
        pl.when(pl.col("time") == DAYS[2])
        .then(17.0)
        .otherwise(pl.col("valuation_price"))
        .alias("valuation_price")
    )
    repair = evaluate_execution(
        transaction.tail(1),
        changed.tail(4),
        checkpoint=prefix.checkpoint,
        prefix_evidence=evidence(DAYS[1], mode="transactions"),
        input_evidence=evidence(DAYS[5], mode="transactions"),
        **args,
    )
    full = evaluate_execution(transaction, changed, **args)
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    before = store.publish_shared("account", "execution", "old", old.tables)
    saved = {
        name: store._table_path(ref["table_id"]).read_bytes()
        for name, ref in before["files"].items()
    }
    refs = store.execution_references(
        repair.tables, previous=before["files"], through=DAYS[1]
    )
    current = store.publish_shared(
        "account",
        "execution",
        "repair",
        {},
        expected_shared=before,
        file_references=refs,
    )
    for name, expected in full.tables.items():
        assert_frame_equal(
            store.read_reference(current["files"][name]),
            expected,
            check_column_order=False,
        )
    assert all(
        store._table_path(before["files"][name]["table_id"]).read_bytes() == raw
        for name, raw in saved.items()
    )
    assert store.shared_receipt("account", "execution", input_identity="old") == before


def test_cancellation_preserves_original_native_checkpoint():
    weights, decisions, prices, calendar = bridge_inputs()
    args = dict(
        initial_capital=100,
        calendar=calendar,
        reference_price_mode="decision",
        config=ExecutionConfig(rate=0, min_fee=0),
    )
    first = run_execution_from_weights(
        weights.head(1),
        decisions.head(3),
        prices.head(3),
        available_date=DAYS[2],
        input_evidence=evidence(DAYS[2]),
        **args,
    )
    before_frames, before_metadata = save_checkpoint(first.checkpoint)
    canceled = False

    def progress(completed, _total):
        nonlocal canceled
        canceled = completed == 1

    def check():
        if canceled:
            raise RuntimeError("canceled")

    with pytest.raises(RuntimeError, match="canceled"):
        run_execution_from_weights(
            weights.tail(1),
            decisions.tail(3),
            prices.tail(3),
            checkpoint=first.checkpoint,
            input_evidence=evidence(DAYS[5]),
            prefix_evidence=evidence(DAYS[2]),
            progress=progress,
            check_canceled=check,
            **args,
        )
    after_frames, after_metadata = save_checkpoint(first.checkpoint)
    assert before_metadata == after_metadata
    for name, frame in before_frames.items():
        assert_frame_equal(frame, after_frames[name])
