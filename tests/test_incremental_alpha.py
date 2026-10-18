import hashlib
import math
from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest
from bagelquant_core import Domain, Node
from bagelquant_core.hashing import hash_dataframe, hash_mapping
from polars.testing import assert_frame_equal

from bagelquant_bt import (
    PERSISTENCE_SUMMARY_POLICY,
    BTStore,
    alpha_input_start,
    evaluate_alpha,
)
from bagelquant_bt.exceptions import InputValidationError
from bagelquant_bt.horizon import SessionWindow


def inputs(count, *, revision=False):
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(count)]
    frame = pl.DataFrame(
        [
            {
                "time": day,
                "asset_id": asset,
                "value": float((index * 7 + number * 3) % 13),
            }
            for index, day in enumerate(dates)
            for number, asset in enumerate(["a", "b", "c", "d"])
        ]
    )
    if revision:
        frame = frame.with_columns(
            pl.when((pl.col("time") == dates[5]) & (pl.col("asset_id") == "a"))
            .then(50.0)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    labels = frame.select("time", "asset_id").with_columns(
        (
            (pl.col("time").cast(pl.Int32) % 7 - 3) * 0.001
            + pl.col("asset_id").replace_strict(
                {"a": 0.001, "b": -0.002, "c": 0.003, "d": -0.001}
            )
        ).alias("forward_return"),
        (pl.col("time") + timedelta(days=1)).alias("interval_start"),
        (pl.col("time") + timedelta(days=2)).alias("interval_end"),
        (pl.col("time") + timedelta(days=2)).alias("available_date"),
    )
    node = Node.from_domain(
        frame, Domain(calendar=dates, universe=["a", "b", "c", "d"])
    )

    def proof(first, last, information_cutoff):
        return hash_mapping(
            {
                "value": node.interval_identity(first, last),
                "labels": hash_dataframe(
                    labels.filter(
                        (pl.col("time") >= first)
                        & (pl.col("time") <= last)
                        & (pl.col("available_date") <= information_cutoff)
                    )
                ),
            }
        )

    return node, labels, pl.DataFrame({"time": dates}), proof


SETTINGS = {
    "windows": (
        SessionWindow("cumulative", "1", 1, 1),
        SessionWindow("cumulative", "3", 1, 3),
        SessionWindow("bucket", "2_5", 2, 5),
    ),
    "persistence_lags": (1, 3, 5),
    "lead_lags": (-3, 0, 2),
    "alpha_return_lags": (0, 2),
    "quantiles": 2,
    "rolling_observations": 5,
}


def test_persistence_append_builds_only_new_pairs_and_one_merged_summary(monkeypatch):
    from bagelquant_bt import evaluation

    alpha, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=("persistence",),
        input_proof=proof,
        **SETTINGS,
    )
    alpha, labels, calendar, proof = inputs(36)
    reference = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=("persistence",),
        input_proof=proof,
        **SETTINGS,
    )
    original_pairs = evaluation._signal_rank_persistence_prepared
    original_summary = evaluation.summarize_signal_persistence
    pair_targets, summary_rows = [], []

    def observe_pairs(*args, **kwargs):
        assert kwargs["target_start"] == date.fromisoformat(previous.state["through"])
        result = original_pairs(*args, **kwargs)
        pair_targets.extend(result["target_date"].to_list())
        return result

    def observe_summary(series, horizons):
        summary_rows.append(series.height)
        return original_summary(series, horizons)

    monkeypatch.setattr(evaluation, "_signal_rank_persistence_prepared", observe_pairs)
    monkeypatch.setattr(evaluation, "summarize_signal_persistence", observe_summary)
    updated = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=("persistence",),
        input_proof=proof,
        previous=previous,
        **SETTINGS,
    )
    assert pair_targets and min(pair_targets) > date.fromisoformat(
        previous.state["through"]
    )
    assert summary_rows == [reference.tables["daily_signal_autocorrelation"].height]
    for name, expected in reference.tables.items():
        assert_frame_equal(updated.tables[name], expected, check_exact=True)
    prior = previous.tables["daily_signal_autocorrelation"]
    retained = updated.tables["daily_signal_autocorrelation"].filter(
        pl.col("target_date") <= date.fromisoformat(previous.state["through"])
    )
    assert_frame_equal(retained, prior, check_exact=True)
    evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=("persistence",),
        input_proof=proof,
        previous=updated,
        **SETTINGS,
    )
    assert summary_rows == [reference.tables["daily_signal_autocorrelation"].height]


@pytest.mark.parametrize(
    "components",
    [("book_tail",), ("turnover",), ("quantiles",), ("lead_lag",), ("alpha_return",)],
)
def test_first_mature_path_restores_original_unprocessed_decisions(components):
    alpha, labels, calendar, _proof = inputs(35)
    # An owner token binds unchanged original labels at each information cutoff.
    def proof(*args):
        return "unchanged-input-prefix"
    cutoff = date(2024, 2, 5)
    labels = labels.with_columns(
        pl.max_horizontal("interval_end", pl.lit(cutoff)).alias("available_date")
    )
    previous = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=date(2024, 2, 4),
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    assert all(value is None for value in previous.state["paths"].values())
    alpha, labels, calendar, _ = inputs(36)
    labels = labels.with_columns(
        pl.max_horizontal("interval_end", pl.lit(cutoff)).alias("available_date")
    )
    assert (
        alpha_input_start(previous, calendar=calendar, components=components)
        == calendar["time"].min()
    )
    updated = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=cutoff,
        components=components,
        previous=previous,
        input_proof=proof,
        **SETTINGS,
    )
    expected = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=cutoff,
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    for name, table in expected.tables.items():
        assert_frame_equal(
            updated.tables[name], table, check_exact=False, rel_tol=1e-12, abs_tol=1e-12
        )


@pytest.mark.parametrize(
    "components", [None, ("persistence",), ("book_tail", "turnover")]
)
def test_bounded_input_consumption_matches_complete_inputs(components):
    first, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        first,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    alpha, labels, calendar, proof = inputs(36)
    first = alpha_input_start(
        previous,
        calendar=calendar,
        components=components,
        **{
            name: SETTINGS[name]
            for name in (
                "windows",
                "persistence_lags",
                "lead_lags",
                "alpha_return_lags",
            )
        },
    )
    assert first > calendar["time"].min()
    selected = alpha.collect(dense=False).filter(pl.col("time") >= first)
    tail = Node.from_domain(selected, alpha.domain)
    updated = evaluate_alpha(
        tail,
        labels.filter(pl.col("time") >= first),
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        previous=previous,
        **SETTINGS,
    )
    reference = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    assert updated.tables.keys() == reference.tables.keys()
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
            updated.tables[name].select(expected.columns).sort(keys),
            expected.sort(keys),
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


@pytest.mark.parametrize(
    "before,after",
    [
        (("persistence",), None),
        (("book_tail",), None),
        (("horizons",), ("horizons", "rolling_ic")),
        (("turnover",), ("turnover", "book_tail")),
        (("rolling_ic",), ("rolling_ic", "horizons")),
    ],
)
@pytest.mark.parametrize("append", [False, True])
def test_component_upgrade_preserves_completed_families(
    before, after, append, monkeypatch
):
    alpha, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=before,
        input_proof=proof,
        **SETTINGS,
    )
    if append:
        alpha, labels, calendar, proof = inputs(36)
    from bagelquant_bt import evaluation

    original = evaluation.simulate_weight_returns
    starts = []

    def observed(weights, returns, **kwargs):
        checkpoint = kwargs.get("checkpoint")
        if checkpoint is not None:
            starts.append((checkpoint.time, returns["time"].min()))
        return original(weights, returns, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "simulate_weight_returns", observed)
        updated = evaluate_alpha(
            alpha,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            components=after,
            previous=previous,
            input_proof=proof,
            **SETTINGS,
        )
    reference = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=after,
        input_proof=proof,
        **SETTINGS,
    )
    assert updated.tables.keys() == reference.tables.keys()
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
            updated.tables[name].select(expected.columns).sort(keys),
            expected.sort(keys),
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    assert all(first is None or first > completed for completed, first in starts)
    assert updated.state["components"] == reference.state["components"]


@pytest.mark.parametrize(
    "components",
    [
        ("horizons", "rolling_ic", "persistence"),
        ("book_tail", "turnover", "quantiles", "lead_lag", "alpha_return"),
        None,
    ],
)
def test_append_reuses_mature_statistics_and_continues_weight_state(
    components, monkeypatch
):
    first, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        first,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    appended, labels, calendar, proof = inputs(36)
    from bagelquant_bt import evaluation

    original_ic = evaluation.window_information_coefficients
    ic_dates = []

    def observed(factor, returns):
        ic_dates.append(factor["evaluation_date"].unique().sort().to_list())
        return original_ic(factor, returns)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "window_information_coefficients", observed)
        updated = evaluate_alpha(
            appended,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            components=components,
            input_proof=proof,
            previous=previous,
            **SETTINGS,
        )
    reference = evaluate_alpha(
        appended,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    assert updated.tables.keys() == reference.tables.keys()
    for name, table in reference.tables.items():
        actual = updated.tables[name]
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
            if column in table.columns
        ]
        assert_frame_equal(
            actual.select(table.columns).sort(keys),
            table.sort(keys),
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    if ic_dates:
        assert all(min(days) >= date(2024, 1, 30) for days in ic_dates if days)
    assert updated.execution["path_sessions"] <= len(updated.state["paths"])
    assert updated.execution["continued_paths"] == sum(
        item is not None for item in previous.state["paths"].values()
    )


def test_changed_prefix_and_missing_proof_are_actionable_before_daily_work(monkeypatch):
    first, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        first,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        **SETTINGS,
    )
    revised, labels, calendar, proof = inputs(36, revision=True)
    monkeypatch.setattr(
        "bagelquant_bt.evaluation.simulate_weight_returns",
        lambda *_a, **_k: pytest.fail("replayed historical paths"),
    )
    with pytest.raises(InputValidationError, match="input prefix changed"):
        evaluate_alpha(
            revised,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            input_proof=proof,
            previous=previous,
            **SETTINGS,
        )


def test_delayed_maturity_resumes_from_actual_path_boundary():
    components = ("book_tail", "turnover", "quantiles", "lead_lag", "alpha_return")

    def build(count):
        node, labels, calendar, _proof = inputs(count)
        labels = labels.with_columns(
            pl.when(pl.col("time") == date(2024, 1, 5))
            .then(pl.lit(date(2024, 2, 5)))
            .otherwise(pl.col("available_date"))
            .alias("available_date")
        )

        def proof(first, last, information_cutoff):
            return hash_mapping(
                {
                    "value": node.interval_identity(first, last),
                    "labels": hash_dataframe(
                        labels.filter(
                            (pl.col("time") >= first)
                            & (pl.col("time") <= last)
                            & (pl.col("available_date") <= information_cutoff)
                        )
                    ),
                }
            )

        return evaluate_alpha(
            node,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            components=components,
            input_proof=proof,
            **SETTINGS,
        ), (node, labels, calendar, proof)

    previous, _ = build(35)
    reference, (node, labels, calendar, proof) = build(36)
    assert previous.state["paths"]["book"]["time"] == "2024-01-04"
    updated = evaluate_alpha(
        node,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        previous=previous,
        **SETTINGS,
    )
    for name, table in reference.tables.items():
        keys = [
            column
            for column in ("time", "lag", "path_kind", "quantile")
            if column in table.columns
        ]
        assert_frame_equal(
            updated.tables[name].sort(keys),
            table.sort(keys),
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_noop_backward_cutoff_and_missing_path_table_do_no_daily_work(monkeypatch):
    from dataclasses import replace

    node, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        node,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        **SETTINGS,
    )
    monkeypatch.setattr(
        "bagelquant_bt.evaluation.simulate_weight_returns",
        lambda *_a, **_k: pytest.fail("ran daily work"),
    )
    current = evaluate_alpha(
        node,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        previous=previous,
        **SETTINGS,
    )
    assert current.execution["outcome"] == "up_to_date"
    with pytest.raises(InputValidationError, match="nonregressing"):
        evaluate_alpha(
            node,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max() - timedelta(days=1),
            input_proof=proof,
            previous=previous,
            **SETTINGS,
        )
    damaged = replace(
        previous,
        tables={
            name: table
            for name, table in previous.tables.items()
            if name != "state_book_returns"
        },
    )
    with pytest.raises(InputValidationError, match="tables are incomplete"):
        evaluate_alpha(
            node,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            input_proof=proof,
            previous=damaged,
            **SETTINGS,
        )
    with pytest.raises(InputValidationError, match="explicit maintenance"):
        evaluate_alpha(
            node,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            previous=previous,
            **SETTINGS,
        )


def _pre_summary_policy(result):
    """Retain a real evaluation's daily/state data with a legacy derived payload.

    The synthetic legacy mean differs by one bit, so a blind previous-result
    return cannot satisfy the current reference. This is not a captured real
    original archive or a claim that every original reduction differs.
    """
    summary = result.tables["horizon_signal_persistence_summary"]
    legacy = summary.with_columns(
        pl.Series(
            "mean_rank_autocorrelation",
            [None if value is None else math.nextafter(value, math.inf)
             for value in summary["mean_rank_autocorrelation"]],
            dtype=pl.Float64,
        )
    )
    return replace(
        result,
        state={key: value for key, value in result.state.items()
               if key != "persistence_summary_policy"},
        tables={**result.tables, "horizon_signal_persistence_summary": legacy},
    )


def _artifact_facts(root):
    result = {}
    for path in root.rglob("*.parquet"):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[path.relative_to(root).as_posix()] = (
            digest.hexdigest(), path.stat().st_mtime_ns
        )
    return result


@pytest.mark.parametrize("old_marker", [None, "bt.persistence.unknown.v0"])
def test_saved_prepolicy_refresh_keeps_daily_objects_receipts_bytes_and_repeat(
    tmp_path, monkeypatch, old_marker
):
    from bagelquant_bt import evaluation

    node, labels, calendar, proof = inputs(35)
    initial = evaluate_alpha(
        node, labels, calendar=calendar, available_date=calendar["time"].max(),
        input_proof=proof, **SETTINGS,
    )
    legacy = _pre_summary_policy(initial)
    if old_marker is not None:
        legacy = replace(legacy, state={**legacy.state,
                                       "persistence_summary_policy": old_marker})
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    old_id = store.begin("alpha", settings={"raw_policy": legacy.state["policy"]},
                         through=legacy.state["through"])
    old_header = store.publish_chapter(
        old_id, "summary", legacy.tables,
        metadata={"metrics": legacy.metrics, "state": legacy.state},
    )
    header, frames = store.read_chapter(old_id, "summary")
    previous = replace(legacy, tables=frames, metrics=header["metrics"],
                       state=header["state"])
    old_facts = _artifact_facts(store.artifact_path)

    def forbidden(*_args, **_kwargs):
        pytest.fail("replayed a daily/account producer during derived refresh")

    calls = []
    summary = evaluation.summarize_signal_persistence

    def observe(frame, horizons):
        calls.append(frame)
        return summary(frame, horizons)

    for name in (
        "_signal_rank_persistence_prepared", "simulate_weight_returns",
        "window_information_coefficients", "window_book_returns",
        "window_tail_returns", "_window_factor_returns_prepared",
        "_window_quantile_forward_returns_with_membership",
        "rolling_window_information_coefficients", "build_statistical_inference",
    ):
        monkeypatch.setattr(evaluation, name, forbidden)
    monkeypatch.setattr(evaluation, "summarize_signal_persistence", observe)
    refreshed = evaluate_alpha(
        node, labels, calendar=calendar, available_date=calendar["time"].max(),
        input_proof=proof, previous=previous, **SETTINGS,
    )
    assert len(calls) == 1
    assert calls[0] is previous.tables["daily_signal_autocorrelation"]
    assert refreshed.state["policy"] == previous.state["policy"]
    assert refreshed.state["paths"] is previous.state["paths"]
    assert refreshed.metrics is previous.metrics
    assert refreshed.checkpoint is previous.checkpoint
    assert refreshed.state["persistence_summary_policy"] == PERSISTENCE_SUMMARY_POLICY
    assert previous.state.get("persistence_summary_policy") == old_marker
    assert {key: value for key, value in refreshed.state.items()
            if key != "persistence_summary_policy"} == {
                key: value for key, value in previous.state.items()
                if key != "persistence_summary_policy"
            }
    assert refreshed.execution == {
        "path_sessions": 0, "continued_paths": 0, "outcome": "derived_refresh"
    }
    for name, frame in previous.tables.items():
        if name == "horizon_signal_persistence_summary":
            assert_frame_equal(refreshed.tables[name], initial.tables[name],
                               check_exact=True)
        else:
            assert refreshed.tables[name] is frame

    new_id = store.begin(
        "alpha", settings={"raw_policy": refreshed.state["policy"],
                           "persistence_summary_policy": PERSISTENCE_SUMMARY_POLICY},
        through=refreshed.state["through"],
    )
    assert new_id != old_id
    unchanged_refs = {
        key: value for key, value in old_header["files"].items()
        if key != "horizon_signal_persistence_summary"
    }
    with pytest.raises(RuntimeError, match="immutable BT chapter already differs"):
        store.publish_chapter(
            old_id, "summary",
            {"horizon_signal_persistence_summary":
             refreshed.tables["horizon_signal_persistence_summary"]},
            file_references=unchanged_refs,
            metadata={"metrics": refreshed.metrics, "state": refreshed.state},
        )
    new_header = store.publish_chapter(
        new_id, "summary",
        {"horizon_signal_persistence_summary":
         refreshed.tables["horizon_signal_persistence_summary"]},
        file_references=unchanged_refs,
        metadata={"metrics": refreshed.metrics, "state": refreshed.state},
    )
    for name, reference in old_header["files"].items():
        if name != "horizon_signal_persistence_summary":
            assert new_header["files"][name] == reference
    assert store.chapter(old_id, "summary", verify=True) == {
        **old_header, "status": "ready"
    }
    after_facts = _artifact_facts(store.artifact_path)
    assert all(after_facts[key] == fact for key, fact in old_facts.items())

    monkeypatch.setattr(evaluation, "summarize_signal_persistence", forbidden)
    repeated = evaluate_alpha(
        node, labels, calendar=calendar, available_date=calendar["time"].max(),
        input_proof=proof, previous=refreshed, **SETTINGS,
    )
    assert repeated.tables is refreshed.tables
    assert repeated.state is refreshed.state
    assert repeated.execution["outcome"] == "up_to_date"


def test_prepolicy_refresh_keeps_prefix_and_cancellation_gates(monkeypatch):
    from bagelquant_bt import evaluation

    node, labels, calendar, proof = inputs(35)
    previous = _pre_summary_policy(evaluate_alpha(
        node, labels, calendar=calendar, available_date=calendar["time"].max(),
        input_proof=proof, **SETTINGS,
    ))
    def forbidden(*_args, **_kwargs):
        pytest.fail("reduced before input-prefix validation")
    monkeypatch.setattr(evaluation, "summarize_signal_persistence", forbidden)
    with pytest.raises(InputValidationError, match="input prefix changed"):
        evaluate_alpha(
            node, labels, calendar=calendar, available_date=calendar["time"].max(),
            input_proof=lambda *_: "changed", previous=previous, **SETTINGS,
        )
    def canceled():
        raise RuntimeError("canceled")
    with pytest.raises(RuntimeError, match="canceled"):
        evaluate_alpha(
            node, labels, calendar=calendar, available_date=calendar["time"].max(),
            input_proof=proof, previous=previous, check_canceled=canceled, **SETTINGS,
        )
    assert "persistence_summary_policy" not in previous.state
    damaged = replace(previous, tables={
        key: value for key, value in previous.tables.items()
        if key != "horizon_signal_persistence_summary"
    })
    with pytest.raises(InputValidationError, match="primitive families are incomplete"):
        evaluate_alpha(
            node, labels, calendar=calendar, available_date=calendar["time"].max(),
            input_proof=proof, previous=damaged, **SETTINGS,
        )


def test_canceled_derived_refresh_does_not_mutate_previous(monkeypatch):
    from bagelquant_bt import evaluation

    node, labels, calendar, proof = inputs(35)
    previous = _pre_summary_policy(evaluate_alpha(
        node, labels, calendar=calendar, available_date=calendar["time"].max(),
        input_proof=proof, **SETTINGS,
    ))
    summary = evaluation.summarize_signal_persistence
    reductions = []
    def observed(*args):
        reductions.append(1)
        return summary(*args)
    monkeypatch.setattr(evaluation, "summarize_signal_persistence", observed)
    checks = []
    def canceled_after_reduction():
        checks.append(1)
        if reductions:
            raise RuntimeError("canceled after summary")
    with pytest.raises(RuntimeError, match="canceled after summary"):
        evaluate_alpha(
            node, labels, calendar=calendar, available_date=calendar["time"].max(),
            input_proof=proof, previous=previous,
            check_canceled=canceled_after_reduction, **SETTINGS,
        )
    assert reductions == [1]
    assert len(checks) >= 3
    assert "persistence_summary_policy" not in previous.state


@pytest.mark.parametrize("repair", [False, True])
def test_prepolicy_append_and_preceding_checkpoint_repair_reuse_daily_work(
    repair, monkeypatch
):
    from bagelquant_bt import evaluation

    old_count = 5 if repair else 35
    node, labels, calendar, proof = inputs(old_count)
    previous = _pre_summary_policy(evaluate_alpha(
        node, labels, calendar=calendar, available_date=calendar["time"].max(),
        input_proof=proof, **SETTINGS,
    ))
    node, labels, calendar, proof = inputs(36, revision=repair)
    reference = evaluate_alpha(
        node, labels, calendar=calendar, available_date=calendar["time"].max(),
        input_proof=proof, **SETTINGS,
    )
    calls = []
    original = evaluation._signal_rank_persistence_prepared
    def observed(*args, **kwargs):
        calls.append(kwargs.get("target_start"))
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluation, "_signal_rank_persistence_prepared", observed)
    updated = evaluate_alpha(
        node, labels, calendar=calendar, available_date=calendar["time"].max(),
        input_proof=proof, previous=previous, **SETTINGS,
    )
    assert calls == [date.fromisoformat(previous.state["through"])]
    assert updated.state["policy"] == previous.state["policy"]
    assert updated.state["persistence_summary_policy"] == PERSISTENCE_SUMMARY_POLICY
    assert updated.execution["path_sessions"] <= (
        (36 - old_count) * len(updated.state["paths"])
    )
    assert updated.tables.keys() == reference.tables.keys()
    for name, expected in reference.tables.items():
        keys = [column for column in (
            "time", "evaluation_date", "target_date", "window_kind", "window_id",
            "lag", "path_kind", "quantile", "horizon_sessions",
        ) if column in expected.columns]
        assert_frame_equal(
            updated.tables[name].select(expected.columns).sort(keys),
            expected.sort(keys), check_exact=False, rel_tol=1e-12, abs_tol=1e-12,
        )
    assert_frame_equal(
        updated.tables["horizon_signal_persistence_summary"],
        reference.tables["horizon_signal_persistence_summary"], check_exact=True,
    )
