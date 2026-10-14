"""Metadata-only affected-suffix planning retains original daily primitives."""

from datetime import date

import polars as pl
import pytest
from bagelquant_core import Node
from bagelquant_core.hashing import hash_dataframe, hash_mapping
from polars.testing import assert_frame_equal

from bagelquant_bt import alpha_update_plan, evaluate_alpha
from test_incremental_alpha import SETTINGS, inputs

PLAN_SETTINGS = {
    name: SETTINGS[name]
    for name in ("windows", "persistence_lags", "lead_lags", "alpha_return_lags")
}


def evaluated(count):
    alpha, labels, calendar, proof = inputs(count)
    result = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        **SETTINGS,
    )
    return result, calendar, proof


def test_repeat_and_append_plan_read_only_metadata(monkeypatch):
    previous, calendar, proof = evaluated(35)
    from bagelquant_bt import evaluation

    def forbidden(*args, **kwargs):
        pytest.fail("planning consumed numerical inputs or ran financial paths")

    monkeypatch.setattr(evaluation, "node_values", forbidden)
    monkeypatch.setattr(evaluation, "simulate_weight_returns", forbidden)
    plan = alpha_update_plan(
        previous.state,
        calendar=calendar,
        available_date=calendar["time"].max(),
        **PLAN_SETTINGS,
    )
    assert plan["status"] == "up_to_date"
    assert plan["read_start"] is None
    _, _, appended, proof = inputs(36)
    plan = alpha_update_plan(
        previous.state,
        calendar=appended,
        available_date=appended["time"].max(),
        input_proof=proof,
        **PLAN_SETTINGS,
    )
    assert plan["status"] == "append"
    assert plan["read_start"] > appended["time"].min()
    assert plan["output_start"] == appended["time"].max()


def test_revision_repairs_suffix_from_proven_preceding_result(monkeypatch):
    current, _, _ = evaluated(35)
    preceding, _, _ = evaluated(12)
    alpha, labels, calendar, _ = inputs(35)
    changed_day = calendar["time"][25]
    frame = alpha.collect(dense=False).with_columns(
        pl.when((pl.col("time") == changed_day) & (pl.col("asset_id") == "a"))
        .then(50.0)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    revised = Node.from_domain(frame, alpha.domain)

    def proof(first, last, cutoff):
        return hash_mapping(
            {
                "value": revised.interval_identity(first, last),
                "labels": hash_dataframe(
                    labels.filter(
                        pl.col("time").is_between(first, last)
                        & (pl.col("available_date") <= cutoff)
                    )
                ),
            }
        )

    plan = alpha_update_plan(
        current.state,
        calendar=calendar,
        available_date=calendar["time"].max(),
        affected_intervals=((changed_day, changed_day),),
        preceding=(preceding.state,),
        input_proof=proof,
        **PLAN_SETTINGS,
    )
    assert plan["status"] == "repair"
    assert plan["previous_index"] == 0
    assert plan["output_start"] == calendar["time"][12]
    from bagelquant_bt import evaluation

    original = evaluation.simulate_weight_returns
    starts = []

    def observed(weights, returns, **kwargs):
        checkpoint = kwargs.get("checkpoint")
        starts.append(
            (None if checkpoint is None else checkpoint.time, returns["time"].min())
        )
        return original(weights, returns, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "simulate_weight_returns", observed)
        repaired = evaluate_alpha(
            revised,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            previous=preceding,
            input_proof=proof,
            **SETTINGS,
        )
    reference = evaluate_alpha(
        revised,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        **SETTINGS,
    )
    assert starts and all(
        first is None or day is None or day > first for first, day in starts
    )
    assert repaired.execution["path_sessions"] < reference.execution["path_sessions"]
    for name, expected in reference.tables.items():
        keys = [
            column
            for column in (
                "time",
                "evaluation_date",
                "target_date",
                "window_kind",
                "window_id",
                "lag",
                "path_kind",
                "quantile",
                "horizon_sessions",
            )
            if column in expected.columns
        ]
        assert_frame_equal(
            repaired.tables[name].select(expected.columns).sort(keys),
            expected.sort(keys),
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_all_history_revision_and_unknown_prefix_produce_blocked_plan():
    current, calendar, _ = evaluated(35)
    plan = alpha_update_plan(
        current.state,
        calendar=calendar,
        available_date=calendar["time"].max(),
        affected_intervals=((date(2024, 1, 1), date(2024, 1, 1)),),
        **PLAN_SETTINGS,
    )
    assert plan["status"] == "blocked"
    assert plan["previous_index"] is None
    assert "preceding" in plan["reason"]
    unknown = alpha_update_plan(
        current.state,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=lambda *_: None,
        **PLAN_SETTINGS,
    )
    assert unknown["status"] == "blocked"


def test_metadata_keeps_arbitrarily_delayed_pending_horizons():
    alpha, labels, calendar, _ = inputs(35)
    delayed = labels.with_columns(pl.lit(date(2024, 3, 1)).alias("available_date"))
    prior = evaluate_alpha(
        alpha,
        delayed,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=lambda *_: "input",
        **SETTINGS,
    )
    assert prior.state["pending_horizon_start"] == str(calendar["time"].min())
    _, _, appended, _ = inputs(36)
    plan = alpha_update_plan(
        prior.state,
        calendar=appended,
        available_date=appended["time"].max(),
        **PLAN_SETTINGS,
    )
    assert plan["read_start"] == appended["time"].min()
