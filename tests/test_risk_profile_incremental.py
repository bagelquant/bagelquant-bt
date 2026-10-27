from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt import (
    RISK_PROFILE_POLICY,
    BTStore,
    RollingRiskProfile,
    plan_risk_profile_update,
    rolling_risk_profile,
)

SETTINGS = dict(regression_window=5, volatility_window=3)


def inputs(count=30):
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(count)]
    rng = np.random.default_rng(81)
    x = rng.normal(0, 0.01, count)
    returns = pl.DataFrame({"time": days, "gross_return": 0.001 + 2 * x})
    factors = pl.DataFrame(
        {
            "time": days[:-1],
            "available_date": days[1:],
            "factor": ["size"] * (count - 1),
            "family": ["style"] * (count - 1),
            "factor_return": x[:-1],
        }
    )
    return days, returns, factors


def proof(first, last, cutoff):
    return f"synthetic:{first}:{last}:{cutoff}"


def test_late_start_catalogue_preserves_original_dates_without_warmup_credit():
    days, returns, factors = inputs(400)
    gate = 30
    catalogue = {
        "size": {"family": "style", "first_seen": str(days[0])},
        "future": {
            "family": "industry",
            "first_seen": str(days[-1] + timedelta(days=1)),
        },
    }
    actual = rolling_risk_profile(
        returns, factors, days[gate:], factor_catalogue=catalogue
    )
    reference = rolling_risk_profile(
        returns.filter(pl.col("time") >= days[gate]),
        factors.filter(pl.col("time") >= days[gate]),
        days[gate:],
    )
    assert actual.state["factors"] == {"size": catalogue["size"]}
    for name in ("returns", "exposures", "diagnostics"):
        assert_frame_equal(
            getattr(actual, name), getattr(reference, name), check_exact=True
        )
    assert actual.exposures["time"].min() >= days[gate + 240]
    with pytest.raises(ValueError, match="precedes its original catalogue"):
        rolling_risk_profile(
            returns,
            factors,
            days[gate:],
            factor_catalogue={
                "size": {"family": "style", "first_seen": str(days[gate + 1])}
            },
        )


def evaluate(days, returns, factors, through, *, previous=None, plan=None):
    selected = days[: days.index(through) + 1]
    plan = plan or plan_risk_profile_update(
        None if previous is None else previous.state,
        calendar=selected,
        through=through,
        input_proof=proof,
        **SETTINGS,
    )
    return rolling_risk_profile(
        returns.filter(pl.col("time").is_between(plan.read_start, through)),
        factors.filter(pl.col("time") >= plan.read_start).filter(
            pl.col("available_date") <= through
        ),
        selected,
        previous=previous,
        plan=plan,
        input_proof=proof,
        previous_input_proof=proof,
        **SETTINGS,
    )


def merged(old, delta, start):
    return RollingRiskProfile(
        *(
            pl.concat(
                [
                    getattr(old, name).filter(pl.col("time") < start),
                    getattr(delta, name),
                ]
            )
            for name in ("returns", "exposures", "diagnostics")
        ),
        state=delta.state,
    )


@pytest.mark.parametrize("minimum", [None, 3])
def test_append_uses_only_input_and_saved_specific_halos_and_counts_solves(
    monkeypatch, minimum,
):
    monkeypatch.setitem(SETTINGS, "regression_min_periods", minimum)
    days, returns, factors = inputs()
    old = evaluate(days, returns, factors, days[17])
    calendar = days[:19]
    plan = plan_risk_profile_update(
        old.state, calendar=calendar, through=days[18], input_proof=proof, **SETTINGS
    )
    assert plan.mode == "append"
    assert plan.read_start == days[12]
    assert plan.output_start == days[17]
    assert plan.specific_start == days[15]
    bounded = replace(
        old,
        returns=old.returns.filter(
            pl.col("time").is_between(
                plan.specific_start, plan.output_start - timedelta(days=1)
            )
        ),
        exposures=old.exposures.clear(),
        diagnostics=old.diagnostics.clear(),
    )
    calls = []
    solve = np.linalg.solve
    monkeypatch.setattr(np.linalg, "solve", lambda *a: (calls.append(1), solve(*a))[1])
    delta = evaluate(days, returns, factors, days[18], previous=bounded, plan=plan)
    assert delta.solve_count == len(calls) == 1
    assert delta.exposures["time"].to_list() == [days[17]]
    complete = merged(old, delta, plan.output_start)
    full = evaluate(days, returns, factors, days[18])
    for name in ("returns", "exposures", "diagnostics"):
        assert_frame_equal(getattr(complete, name), getattr(full, name))
    assert complete.state == full.state
    assert complete.state["policy"]["kernel"] == RISK_PROFILE_POLICY


def test_new_industry_never_changes_historical_population_or_diagnostics():
    days, returns, factors = inputs()
    industry = pl.DataFrame(
        {
            "time": days[18:-1],
            "available_date": days[19:],
            "factor": ["industry:new"] * (len(days) - 19),
            "family": ["industry"] * (len(days) - 19),
            "factor_return": [0.01] * (len(days) - 19),
        }
    )
    factors = pl.concat([factors, industry])
    old = evaluate(days, returns, factors, days[18])
    delta = evaluate(days, returns, factors, days[-1], previous=old)
    full = evaluate(days, returns, factors, days[-1])
    assert old.exposures.filter(pl.col("factor") == "industry:new").is_empty()
    assert_frame_equal(old.exposures, full.exposures.filter(pl.col("time") < days[18]))
    assert_frame_equal(
        old.diagnostics.filter(pl.col("time") < days[18]),
        full.diagnostics.filter(pl.col("time") < days[18]),
    )
    complete = merged(old, delta, days[18])
    for name in ("returns", "exposures", "diagnostics"):
        assert_frame_equal(getattr(complete, name), getattr(full, name))


def test_metadata_repeat_and_missing_or_changed_proofs_never_run_regressions(
    monkeypatch,
):
    days, returns, factors = inputs()
    old = evaluate(days, returns, factors, days[18])

    def forbidden(*_):
        raise AssertionError("metadata planning/repeat solved old history")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    plan = plan_risk_profile_update(
        old.state, calendar=days[:19], through=days[18], input_proof=proof, **SETTINGS
    )
    assert plan.complete
    empty = rolling_risk_profile(
        returns.clear(),
        factors.clear(),
        days[:19],
        previous=old,
        plan=plan,
        input_proof=proof,
        **SETTINGS,
    )
    assert empty.solve_count == 0 and empty.returns.is_empty()
    assert empty.state == old.state
    for state, token in (
        ({}, proof),
        (old.state, lambda *_: "changed"),
        (old.state, None),
    ):
        blocked = plan_risk_profile_update(
            state, calendar=days[:20], through=days[19], input_proof=token, **SETTINGS
        )
        assert blocked.mode == "blocked"
    blocked = plan_risk_profile_update(
        old.state,
        calendar=days[:20],
        through=days[19],
        input_proof=proof,
        affected_intervals=[(days[10], days[10])],
        **SETTINGS,
    )
    assert blocked.mode == "blocked"


@pytest.mark.parametrize("minimum", [None, 3])
def test_affected_repair_selects_proven_original_state_and_matches_full(
    monkeypatch, minimum,
):
    monkeypatch.setitem(SETTINGS, "regression_min_periods", minimum)
    days, returns, factors = inputs()
    early = evaluate(days, returns, factors, days[9])
    latest = evaluate(days, returns, factors, days[18])
    changed = returns.with_columns(
        pl.when(pl.col("time") == days[10])
        .then(0.2)
        .otherwise(pl.col("gross_return"))
        .alias("gross_return")
    )
    plan = plan_risk_profile_update(
        latest.state,
        calendar=days[:19],
        through=days[18],
        affected_intervals=[(days[10], days[10])],
        preceding=[early.state],
        input_proof=proof,
        **SETTINGS,
    )
    assert plan.mode == "repair" and plan.previous_index == 0
    repaired = evaluate(
        days,
        changed,
        factors,
        days[18],
        previous=replace(early, returns=latest.returns),
        plan=plan,
    )
    full = evaluate(days, changed, factors, days[18])
    complete = RollingRiskProfile(
        *(
            pl.concat(
                [
                    getattr(latest, name).filter(pl.col("time") < plan.output_start),
                    getattr(repaired, name),
                    getattr(latest, name).filter(pl.col("time") > plan.output_end),
                ]
            )
            for name in ("returns", "exposures", "diagnostics")
        ),
        state=repaired.state,
    )
    assert plan.output_end == days[17]
    assert repaired.solve_count == 8
    assert repaired.state == full.state
    for name in ("returns", "exposures", "diagnostics"):
        assert_frame_equal(getattr(complete, name), getattr(full, name))


def test_halo_loss_and_cancellation_preserve_published_originals(tmp_path):
    days, returns, factors = inputs()
    old = evaluate(days, returns, factors, days[18])
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    original = store.publish_table(old.returns)
    with pytest.raises(ValueError, match="halo is incomplete"):
        evaluate(
            days,
            returns,
            factors,
            days[19],
            previous=replace(old, returns=old.returns.clear()),
        )

    def cancel():
        raise RuntimeError("canceled")

    with pytest.raises(RuntimeError, match="canceled"):
        rolling_risk_profile(
            returns, factors, days, input_proof=proof, check_canceled=cancel, **SETTINGS
        )
    assert_frame_equal(store.read_reference(original), old.returns)


def test_delta_publication_preserves_original_files_without_reading_history(
    tmp_path, monkeypatch
):
    days, returns, factors = inputs()
    old = evaluate(days, returns, factors, days[18])
    delta = evaluate(days, returns, factors, days[19], previous=old)
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    original = {
        name: store.publish_table(getattr(old, name))
        for name in ("returns", "exposures", "diagnostics")
    }
    read = BTStore.read_reference
    monkeypatch.setattr(
        BTStore,
        "read_reference",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("delta publication read numerical history")
        ),
    )
    references = store.publish_continuation(
        {name: getattr(delta, name) for name in original},
        previous=original,
        output_start=days[18],
    )
    monkeypatch.setattr(BTStore, "read_reference", read)
    full = evaluate(days, returns, factors, days[19])
    for name in original:
        assert_frame_equal(store.read_reference(references[name]), getattr(full, name))
        assert store._table_path(original[name]["table_id"]).is_file()
