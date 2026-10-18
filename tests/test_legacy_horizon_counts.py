"""Persisted incomplete-window count schemas remain usable without replay."""

import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt import BTStore, EvaluationResult, alpha_update_plan, evaluate_alpha
from bagelquant_bt.exceptions import InputValidationError
from test_incremental_alpha import inputs


def _saved_files(store):
    return {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in store.artifact_path.rglob("*.parquet")
    }


@pytest.fixture(scope="module")
def saved_legacy(tmp_path_factory):
    root = tmp_path_factory.mktemp("legacy_horizon_counts")
    store = BTStore(root / "meta.sqlite", root / "artifacts")
    store.initialize()
    for count in (64, 256):
        value, labels, calendar, proof = inputs(count)
        result = evaluate_alpha(
            value,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            input_proof=proof,
        )
        tables = dict(result.tables)
        if count == 64:
            # Exact public aggregate representation emitted by the original
            # implementation when 120-session windows are still incomplete:
            # their Int64 empty schemas promoted populated shorter windows.
            for name in (
                "horizon_book_returns",
                "horizon_tail_returns",
                "horizon_quantile_forward_returns",
                "horizon_factor_returns",
            ):
                frame = tables[name]
                tables[name] = frame.cast(
                    {
                        column: pl.Int64
                        for column in (
                            "expected_count",
                            "observed_count",
                            "sample_size",
                        )
                        if column in frame
                    }
                )
        store.publish_shared(
            "legacy",
            f"prefix{count}",
            "original",
            tables,
            metadata={"state": result.state, "metrics": result.metrics},
        )
    return store


def _load(store, count):
    receipt, tables = store.shared_evaluation("legacy", f"prefix{count}")
    return receipt, EvaluationResult(
        tables, receipt["metadata"]["metrics"], state=receipt["metadata"]["state"]
    )


def _metrics_close(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _metrics_close(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _metrics_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


@pytest.mark.parametrize("stage", ["append", "repair"])
def test_persisted_legacy_counts_match_full_reference_and_reuse_daily_checkpoints(
    saved_legacy, monkeypatch, stage
):
    store = saved_legacy
    original, previous = _load(store, 64)
    _, current = _load(store, 256)
    before = _saved_files(store)
    count = 66 if stage == "append" else 258
    value, labels, calendar, proof = inputs(count)
    changed_day = date(2024, 1, 1) + timedelta(days=248)
    if stage == "repair":
        from bagelquant_core import Node
        from bagelquant_core.hashing import hash_dataframe, hash_mapping

        revised = value.collect(dense=False).with_columns(
            pl.when((pl.col("time") == changed_day) & (pl.col("asset_id") == "a"))
            .then(50.0)
            .otherwise(pl.col("value"))
            .alias("value")
        )
        value = Node.from_domain(revised, value.domain)

        def proof(first, last, cutoff):
            return hash_mapping(
                {
                    "value": value.interval_identity(first, last),
                    "labels": hash_dataframe(
                        labels.filter(
                            pl.col("time").is_between(first, last)
                            & (pl.col("available_date") <= cutoff)
                        )
                    ),
                }
            )

    plan = alpha_update_plan(
        (previous if stage == "append" else current).state,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        affected_intervals=() if stage == "append" else ((changed_day, changed_day),),
        preceding=(previous.state,),
    )
    assert plan["status"] == stage
    from bagelquant_bt import evaluation

    simulate = evaluation.simulate_weight_returns
    calls = []

    def observed(weights, returns, **kwargs):
        checkpoint = kwargs.get("checkpoint")
        calls.append(checkpoint)
        assert (
            checkpoint is None
            or returns.is_empty()
            or returns["time"].min() > checkpoint.time
        )
        return simulate(weights, returns, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "simulate_weight_returns", observed)
        actual = evaluate_alpha(
            value,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            input_proof=proof,
            previous=previous,
        )
    expected = evaluate_alpha(
        value,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
    )
    assert len(actual.tables) == len(expected.tables) == 280
    assert actual.tables.keys() == expected.tables.keys()
    for name, frame in expected.tables.items():
        assert_frame_equal(actual.tables[name], frame, rel_tol=1e-12, abs_tol=1e-12)
    assert _metrics_close(actual.metrics, expected.metrics)
    assert actual.state == expected.state
    assert calls and actual.execution["continued_paths"] == sum(
        item is not None for item in calls
    )
    assert actual.execution["continued_paths"] > 0
    assert actual.execution["path_sessions"] < expected.execution["path_sessions"]
    if stage == "append":
        assert actual.execution["path_sessions"] <= 2 * len(actual.state["paths"])
    assert store.shared_receipt("legacy", "prefix64") == original
    assert _saved_files(store) == before


def test_actual_original_empty_ic_schema_gains_sample_size_without_daily_replay(
    tmp_path,
):
    fixture = Path(__file__).parent / "fixtures" / "legacy_empty_horizon_ic.parquet"
    provenance = json.loads(fixture.with_suffix(".json").read_text())
    assert provenance["producer_bt_head"] == "247e32c256981953c514854f223db2c1ba359826"
    assert (
        hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    )
    original_ic = pl.read_parquet(fixture)
    assert original_ic.is_empty() and "sample_size" not in original_ic
    value, labels, calendar, proof = inputs(1)
    original = evaluate_alpha(
        value,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
    )
    tables = {**original.tables, "horizon_ic": original_ic}
    for name, metadata in provenance["additional_fixture_frames"].items():
        path = fixture.parent / metadata["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
        tables[name] = pl.read_parquet(path)
        assert tables[name].is_empty()
    store = BTStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    saved = store.publish_shared(
        "legacy",
        "empty",
        "original",
        tables,
        metadata={"state": original.state, "metrics": original.metrics},
    )
    receipt, tables = store.shared_evaluation("legacy", "empty")
    previous = EvaluationResult(
        tables, receipt["metadata"]["metrics"], state=receipt["metadata"]["state"]
    )
    before = _saved_files(store)
    value, labels, calendar, proof = inputs(2)
    actual = evaluate_alpha(
        value,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        previous=previous,
    )
    expected = evaluate_alpha(
        value,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
    )
    assert actual.tables.keys() == expected.tables.keys()
    for name, frame in expected.tables.items():
        assert_frame_equal(actual.tables[name], frame, rel_tol=1e-12, abs_tol=1e-12)
    assert actual.tables["horizon_ic"].schema["sample_size"] == pl.Int64
    assert actual.execution["path_sessions"] == actual.execution["continued_paths"] == 0
    assert store.shared_receipt("legacy", "empty") == saved
    assert _saved_files(store) == before
    references = store.publish_continuation(actual.tables, previous=receipt["files"])
    store.publish_shared(
        "legacy",
        "empty_appended",
        "new",
        {},
        file_references=references,
        metadata={"state": actual.state, "metrics": actual.metrics},
    )
    _, reloaded = store.shared_evaluation("legacy", "empty_appended")
    for name, frame in actual.tables.items():
        assert_frame_equal(reloaded[name], frame, check_exact=True)
    assert store.shared_receipt("legacy", "empty") == saved
    after = _saved_files(store)
    assert all(after[path] == evidence for path, evidence in before.items())


@pytest.mark.parametrize("values", [[], [1]])
def test_continuation_persists_count_dtype_changes_when_values_are_equal(
    tmp_path, values
):
    store = BTStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    old = pl.DataFrame({"expected_count": pl.Series(values, dtype=pl.Int64)})
    canonical = old.cast({"expected_count": pl.UInt32})
    assert old.equals(canonical) and old.schema != canonical.schema
    original = store.publish_table(old)
    before = _saved_files(store)
    references = store.publish_continuation(
        {"counts": canonical}, previous={"counts": original}
    )
    assert references["counts"]["table_id"] != original["table_id"]
    assert_frame_equal(
        store.read_reference(references["counts"]), canonical, check_exact=True
    )
    after = _saved_files(store)
    assert all(after[path] == evidence for path, evidence in before.items())


@pytest.mark.parametrize(
    "impossible,dtype,error,reason",
    [
        (-1, pl.Int64, pl.exceptions.InvalidOperationError, "conversion"),
        (2**32, pl.Int64, pl.exceptions.InvalidOperationError, "conversion"),
        (1.5, pl.Float64, InputValidationError, "count dtype"),
        ("1", pl.String, InputValidationError, "count dtype"),
    ],
)
def test_legacy_count_normalization_rejects_invalid_counts(
    saved_legacy, impossible, dtype, error, reason
):
    _, previous = _load(saved_legacy, 64)
    tables = dict(previous.tables)
    tables["horizon_book_returns"] = tables["horizon_book_returns"].with_columns(
        pl.lit(impossible, dtype=dtype).alias("expected_count")
    )
    invalid = EvaluationResult(tables, previous.metrics, state=previous.state)
    value, labels, calendar, proof = inputs(66)
    with pytest.raises(error, match=reason):
        evaluate_alpha(
            value,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            input_proof=proof,
            previous=invalid,
        )
