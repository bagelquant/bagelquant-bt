from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt import (
    BTStore,
    plan_risk_profile_update,
    rolling_risk_profile,
    table_composition,
)


def inputs():
    # Four effective economic sessions; their first signal belongs to a retained
    # preceding session. Source time is the signal, never the economic start.
    days = [date(2024, 1, 2) + timedelta(days=i) for i in range(4)]
    gross = pl.DataFrame(
        {
            "time": [day - timedelta(days=1) for day in days[:-1]],
            "interval_start": days[:-1],
            "interval_end": days[1:],
            "available_date": days[1:],
            "gross_return": [0.03, 0.05, 0.07],
        }
    )
    factors = pl.DataFrame(
        {
            "decision_date": gross["time"],
            "time": days[:-1],
            "available_date": days[1:],
            "factor": ["size"] * 3,
            "family": ["style"] * 3,
            "factor_return": [0.01, 0.02, 0.03],
        }
    )
    return days, gross, factors


def test_economic_interval_proof_selection_and_one_solve_match_four_session_reference(
    tmp_path, monkeypatch
):
    days, gross, factors = inputs()
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    gross_ref, factor_ref = store.publish_table(gross), store.publish_table(factors)
    assert set(
        store.reference_evidence(gross_ref, time_column="interval_start")["dates"]
    ) == {str(day) for day in days[:-1]}
    bounded = store.read_reference(
        gross_ref,
        filters=[
            {
                "column": "interval_start",
                "operation": "eq",
                "value": days[1],
            }
        ],
    )
    assert bounded["time"].to_list() == [days[0]]
    assert bounded["interval_start"].to_list() == [days[1]]

    def proof(first, last, cutoff):
        return str(
            (
                store.reference_interval_identity(
                    gross_ref,
                    start=first,
                    end=last,
                    cutoff=cutoff,
                    time_column="interval_start",
                ),
                store.reference_interval_identity(
                    factor_ref, start=first, end=last, cutoff=cutoff
                ),
            )
        )

    def evaluate(through, previous=None):
        calendar = [day for day in days if day <= through]
        plan = plan_risk_profile_update(
            None if previous is None else previous.state,
            calendar=calendar,
            through=through,
            input_proof=proof,
            regression_window=2,
            volatility_window=2,
        )
        # This explicit caller mapping preserves interval/availability columns.
        aligned = (
            gross.filter(pl.col("available_date") <= through)
            .with_columns(
                pl.col("time").alias("signal_time"),
                pl.col("interval_start").alias("time"),
            )
            .filter(pl.col("time").is_between(plan.read_start, plan.read_end))
        )
        return rolling_risk_profile(
            aligned,
            factors.filter(pl.col("available_date") <= through),
            calendar,
            previous=previous,
            plan=plan,
            input_proof=proof,
            regression_window=2,
            volatility_window=2,
        )

    original = evaluate(days[2])
    calls, solve = [], np.linalg.solve
    monkeypatch.setattr(np.linalg, "solve", lambda *a: (calls.append(1), solve(*a))[1])
    delta = evaluate(days[3], original)
    assert delta.solve_count == len(calls) == 1
    beta = delta.exposures["beta"][0]
    # x=.01,.02 -> standardized -1,+1; y=.03,.05 -> centered -.01,+.01.
    # Ridge 1 yields original-unit beta=(.02/(2+1))/.005=4/3.
    assert beta == pytest.approx(4 / 3)
    assert delta.exposures["time"].to_list() == [days[2]]
    assert delta.exposures["contribution"][0] == pytest.approx(0.04)
    assert delta.returns["specific_return"][0] == pytest.approx(0.03)
    reference = evaluate(days[3])
    for name in ("returns", "exposures", "diagnostics"):
        combined = pl.concat(
            [
                getattr(original, name).filter(pl.col("time") < days[2]),
                getattr(delta, name),
            ]
        )
        assert_frame_equal(combined, getattr(reference, name))


def test_economic_axis_missing_from_legacy_index_is_unknown_until_explicit_maintenance(
    tmp_path, monkeypatch
):
    days, gross, _ = inputs()
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    receipt = store.publish_shared("alpha", "gross", "original", {"gross": gross})
    ref = receipt["files"]["gross"]
    before = store._table_path(ref["table_id"]).read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(
            table_composition,
            "COORDINATES",
            tuple(
                name
                for name in table_composition.COORDINATES
                if name not in {"interval_start", "interval_end"}
            ),
        )
        legacy = store._index_schema(ref, gross.schema, frame=gross)
    with store._connect() as connection:
        connection.execute(
            "UPDATE bt_table_index SET schema_json=? WHERE table_id=?",
            (legacy, ref["table_id"]),
        )
    store.verify_reference(ref)
    with monkeypatch.context() as patch:
        patch.setattr(
            BTStore,
            "read_reference",
            lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("metadata proof read history")
            ),
        )
        assert store.reference_evidence(ref) is not None
        assert store.reference_evidence(ref, time_column="interval_start") is None
        assert (
            store.reference_interval_identity(
                ref, start=days[0], end=days[2], time_column="interval_start"
            )
            is None
        )
        plan = store.index_plan(
            evidence=True, references=[ref], time_column="interval_start"
        )
        assert len(plan["tables"]) == 1
    assert store.build_index(plan)["status"] == "complete"
    assert (
        store.reference_interval_identity(
            ref, start=days[0], end=days[2], time_column="interval_start"
        )
        is not None
    )
    assert store._table_path(ref["table_id"]).read_bytes() == before
    assert store.shared_receipt("alpha", "gross", input_identity="original") == receipt
    assert store.reference_evidence(ref, time_column="missing_axis") is None
