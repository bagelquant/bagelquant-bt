"""Standalone immutable BT storage and publication failure evidence."""

import json
from datetime import date

import polars as pl
import pytest

from bagelquant_bt.store import BTStore, evaluation_identity


def _store(tmp_path):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    return store


def _frame(values=(1.0, 2.0)):
    return pl.DataFrame(
        {
            "time": [date(2024, 1, 2), date(2024, 1, 3)],
            "asset_id": ["A", "B"],
            "value": values,
        }
    )


def test_metadata_and_selected_reads_share_only_finite_file_proofs(
    tmp_path, monkeypatch
):
    import bagelquant_bt.store as storage

    store = _store(tmp_path)
    evaluation = store.begin("root", settings={}, through="2024-01-03")
    store.publish_chapter(
        evaluation,
        "summary",
        {
            "first": _frame(),
            "same": _frame(),
            "other": _frame((3.0, 4.0)),
        },
    )
    hashes = []
    original = storage._file_hash

    def checksum(path):
        hashes.append(path)
        return original(path)

    monkeypatch.setattr(storage, "_file_hash", checksum)
    with store._read_connection() as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA query_only").fetchone()[0] == 1
    with store._connect() as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("DELETE FROM bt_chapters")
        assert store.chapter(evaluation, "summary", verify=False)["status"] == "ready"
        writer.rollback()
    assert hashes == []
    with store.read_context() as reader:
        _, frames = reader.read_chapter(evaluation, "summary", names=["first", "same"])
        assert frames["first"].equals(frames["same"])
        reader.read_chapter(evaluation, "summary", names=["first"])
        assert len(hashes) == 0
    reader.read_chapter(evaluation, "summary", names=["first"])
    assert len(hashes) == 0


@pytest.mark.parametrize("kind", ["evaluation", "chapter", "shared"])
def test_receipt_payload_corruption_is_detected_without_rewriting_artifacts(
    tmp_path, kind
):
    store = _store(tmp_path)
    evaluation = store.begin("root", settings={"rate": 1}, through="2024-01-03")
    store.publish_chapter(
        evaluation, "summary", {"rows": _frame()}, metadata={"metrics": {"sharpe": 1}}
    )
    store.publish_shared(
        "root",
        "account",
        "prefix",
        {"rows": _frame()},
        metadata={"through": "2024-01-03"},
    )
    table = {
        "evaluation": "bt_evaluations",
        "chapter": "bt_chapters",
        "shared": "bt_shared",
    }[kind]
    with store._connect() as db:
        row = db.execute(f"SELECT manifest_json FROM {table}").fetchone()
        record = json.loads(row[0])
        if kind == "evaluation":
            record["settings"]["rate"] = 999
        elif kind == "chapter":
            record["metrics"]["sharpe"] = 999
        else:
            record["metadata"]["through"] = "2025-01-01"
        db.execute(f"UPDATE {table} SET manifest_json=?", (json.dumps(record),))
    assert store.verify()["status"] == "corrupt"
    assert store.verify()["issues"][0]["kind"] == "receipt"
    assert store.recover()["unreferenced"] == []
    with pytest.raises(ValueError, match="identity"):
        store.cleanup_plan()
    if kind == "chapter":
        assert store.chapter(evaluation, "summary")["status"] == "invalid"
    else:
        with pytest.raises(ValueError, match="identity"):
            store.describe(
                evaluation
            ) if kind == "evaluation" else store.shared_receipt("root", "account")


def test_shared_json_round_trip_and_identical_retry_are_stable(tmp_path):
    store = _store(tmp_path)
    first = store.publish_shared(
        "root",
        "account",
        "prefix",
        {"rows": _frame()},
        metadata={"through": date(2024, 1, 3)},
    )
    assert first == store.shared_receipt("root", "account")
    assert first == store.publish_shared(
        "root",
        "account",
        "prefix",
        {"rows": _frame()},
        metadata={"through": date(2024, 1, 3)},
    )
    later = store.publish_shared(
        "root", "account", "append", {"rows": _frame((3.0, 4.0))}, expected_shared=first
    )
    assert later == store.shared_receipt("root", "account")


def test_begin_hashes_every_frame_row_and_rejects_arbitrary_objects(tmp_path):
    store = _store(tmp_path)
    first = pl.DataFrame({"time": range(100), "value": range(100)})
    changed = first.with_columns(
        pl.when(pl.col("time") == 50)
        .then(999)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    a = store.begin(
        "root", settings={}, through="2024-01-03", inputs={"returns": first}
    )
    b = store.begin(
        "root", settings={}, through="2024-01-03", inputs={"returns": changed}
    )
    c = store.begin(
        "root", settings={}, through="2024-01-03", inputs={"returns": first.reverse()}
    )
    assert a != b
    assert a == c
    assert store.describe(a)["inputs"]["returns"]["table"]
    with pytest.raises(TypeError, match="JSON"):
        store.begin(
            "root", settings={}, through="2024-01-03", inputs={"arbitrary": object()}
        )


def test_chapter_rejects_reserved_identity_metadata(tmp_path):
    store = _store(tmp_path)
    evaluation = store.begin("root", settings={}, through="2024-01-03")
    with pytest.raises(ValueError, match="reserved"):
        store.publish_chapter(
            evaluation, "summary", {}, metadata={"updated_at": "caller"}
        )


def test_integrity_checks_historical_checksums_for_identical_logical_rows(tmp_path):
    store = _store(tmp_path)
    original = store.publish_shared("root", "account", "old", {"rows": _frame()})
    reference = original["files"]["rows"]
    path = store.artifact_path / "tables" / (reference["table_id"] + ".parquet")
    _frame().write_parquet(path, compression="zstd")
    latest = store.publish_shared(
        "root", "account", "new", {"rows": _frame()}, expected_shared=original
    )
    assert latest["files"]["rows"]["table_id"] == reference["table_id"]
    assert latest["files"]["rows"]["checksum"] == reference["checksum"]
    assert store.read_reference(latest["files"]["rows"]).equals(_frame())
    integrity = store.verify()
    assert integrity["tables"] == 1
    assert integrity["status"] == "corrupt"
    assert integrity["issues"][0]["table_id"] == reference["table_id"]


def test_standalone_history_opaque_refs_and_chapter_deduplication(tmp_path):
    store = _store(tmp_path)
    evaluation = store.begin(
        "materialization",
        settings={"rate": 0.0005},
        through=date(2024, 1, 3),
        mode="weights",
    )
    shared = store.publish_shared(
        "materialization", "weights", "prefix", {"path": _frame()}
    )
    for section in ("summary", "performance"):
        receipt = store.publish_chapter(
            evaluation, section, {"path": _frame()}, metadata={"metrics": {"n": 2}}
        )
        assert receipt["files"]["path"] == shared["files"]["path"]
        assert "path" not in receipt["files"]["path"]
    restarted = BTStore(store.meta_path, store.artifact_path)
    restarted.initialize()
    assert restarted.history("materialization")[0]["id"] == evaluation
    assert restarted.select(evaluation)["mode"] == "weights"
    assert restarted.read_chapter(evaluation, "summary")[1]["path"].equals(_frame())
    assert len(restarted.inventory()["tables"]) == 1
    assert restarted.verify()["status"] == "complete"


def test_shared_cas_cancel_and_immutable_chapter_keep_prior_result(tmp_path):
    store = _store(tmp_path)
    first = store.publish_shared("root", "account", "old", {"path": _frame()})
    with pytest.raises(RuntimeError, match="changed during publication"):
        store.publish_shared("root", "account", "next", {"path": _frame((3.0, 4.0))})
    assert store.shared_receipt("root", "account") == first
    evaluation = store.begin("root", settings={}, through="2024-01-03")
    published = store.publish_chapter(evaluation, "summary", {"path": _frame()})
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise InterruptedError("cancel")

    with pytest.raises(InterruptedError):
        store.publish_chapter(
            evaluation, "summary", {"path": _frame((5.0, 6.0))}, check_canceled=cancel
        )
    assert (
        store.chapter(evaluation, "summary")["receipt_identity"]
        == published["receipt_identity"]
    )
    with pytest.raises(RuntimeError, match="already differs"):
        store.publish_chapter(evaluation, "summary", {"path": _frame((3.0, 4.0))})
    assert store.read_chapter(evaluation, "summary")[1]["path"].equals(_frame())


def test_selective_read_defers_unselected_corruption_to_audit(tmp_path):
    store = _store(tmp_path)
    evaluation = store.begin("root", settings={}, through="2024-01-03")
    receipt = store.publish_chapter(
        evaluation,
        "summary",
        {
            "selected": _frame(),
            "unselected": _frame((3.0, 4.0)),
        },
    )
    ref = receipt["files"]["unselected"]
    (store.artifact_path / "tables" / (ref["table_id"] + ".parquet")).write_bytes(
        b"damage"
    )
    _, frames = store.read_chapter(evaluation, "summary", names=["selected"])
    assert frames["selected"].equals(_frame())
    assert store.chapter(evaluation, "summary", verify=False)["status"] == "ready"
    with pytest.raises(ValueError, match=r"corrupt|artifact"):
        store.read_chapter(evaluation, "summary", names=["unselected"])
    assert store.verify()["status"] == "corrupt"
    assert store.invalidate(reason="source_changed", owner_id="root") == 1
    assert store.chapter(evaluation, "summary")["reason"] == "source_changed"
    with pytest.raises(ValueError, match="source_changed"):
        store.read_chapter(evaluation, "summary", names=["selected"])


def test_recovery_never_adopts_orphans_and_cleanup_is_frozen_idempotent(tmp_path):
    store = _store(tmp_path)
    retained = store.publish_shared("root", "path", "old", {"path": _frame()})
    with pytest.raises(RuntimeError):
        store.publish_shared("root", "path", "new", {"path": _frame((3.0, 4.0))})
    plan = store.cleanup_plan()
    assert len(plan["candidates"]) == 1
    assert store.recover()["unreferenced"] == plan["candidates"]
    assert store.shared_receipt("root", "path") == retained
    tampered = {**plan, "candidates": []}
    with pytest.raises(ValueError, match="identity differs"):
        store.cleanup(tampered)
    assert store.cleanup(plan)["deleted"] == 1
    assert store.cleanup(plan)["deleted"] == 1
    assert store.cleanup_plan()["candidates"] == []
    assert store.verify()["status"] == "complete"


def test_cleanup_rechecks_new_reference_and_protects_unknown_files(tmp_path):
    store = _store(tmp_path)
    store.publish_shared("root", "path", "old", {"path": _frame()})
    with pytest.raises(RuntimeError):
        store.publish_shared("root", "path", "new", {"path": _frame((3.0, 4.0))})
    plan = store.cleanup_plan()
    unknown = store.artifact_path / "tables" / "user-authored.parquet"
    unknown.write_bytes(b"preserve")
    previous = store.shared_receipt("root", "path")
    store.publish_shared(
        "root", "path", "new", {"path": _frame((3.0, 4.0))}, expected_shared=previous
    )
    with pytest.raises(RuntimeError, match="inventory changed"):
        store.cleanup(plan)
    assert unknown.read_bytes() == b"preserve"
    assert store.verify()["status"] == "complete"


def test_public_references_reject_path_escape_and_wrong_logical_proof(tmp_path):
    store = _store(tmp_path)
    receipt = store.publish_shared("root", "path", "old", {"path": _frame()})
    reference = receipt["files"]["path"]
    with pytest.raises(ValueError, match="invalid BT table"):
        store.read_reference({**reference, "table_id": "../outside"})
    with pytest.raises(ValueError, match="logical identity"):
        store.read_reference({**reference, "logical_hash": "wrong"})


def test_input_identity_is_canonical_and_changes_with_numerical_policy():
    inputs = {"weights": _frame(), "receipt": {"identity": "root"}}
    first = evaluation_identity(
        "weights", inputs=inputs, settings={"lag": 1}, kernel="v1"
    )
    shuffled = {
        **inputs,
        "weights": _frame().reverse().select("value", "asset_id", "time"),
    }
    assert (
        evaluation_identity(
            "weights", inputs=shuffled, settings={"lag": 1}, kernel="v1"
        )
        == first
    )
    assert (
        evaluation_identity("weights", inputs=inputs, settings={"lag": 2}, kernel="v1")
        != first
    )
    assert (
        evaluation_identity("weights", inputs=inputs, settings={"lag": 1}, kernel="v2")
        != first
    )


def test_passive_inventory_does_not_create_a_database_or_artifacts(tmp_path):
    store = BTStore(tmp_path / "missing.sqlite", tmp_path / "missing-artifacts")
    assert store.inventory()["tables"] == []
    assert store.describe("missing") is None
    assert store.history("missing") == []
    assert store.shared_receipt("root", "component") is None
    assert store.chapter("missing", "summary")["status"] == "missing"
    assert store.cleanup_plan()["candidates"] == []
    assert list(tmp_path.iterdir()) == []


def test_auxiliary_receipts_participate_in_evaluation_identity(tmp_path):
    store = _store(tmp_path)
    first = store.begin(
        "root", settings={}, through="2024-01-03", inputs={"market": "generation-1"}
    )
    assert (
        store.begin(
            "root", settings={}, through="2024-01-03", inputs={"market": "generation-1"}
        )
        == first
    )
    assert (
        store.begin(
            "root", settings={}, through="2024-01-03", inputs={"market": "generation-2"}
        )
        != first
    )


def test_table_directory_links_are_not_followed(tmp_path):
    store = _store(tmp_path)
    destination = tmp_path / "outside"
    destination.mkdir()
    linked = store.artifact_path / "tables"
    try:
        linked.symlink_to(destination, target_is_directory=True)
    except OSError:
        pytest.skip("link creation unavailable")
    with pytest.raises(ValueError, match="escapes"):
        store.read_reference({"backend": "bt", "table_id": "a" * 64})
    with pytest.raises(ValueError, match="symbolic link"):
        store.cleanup_plan()
    assert destination.exists()


def test_interrupted_cleanup_resumes_exact_original_inventory(tmp_path):
    store = _store(tmp_path)
    store.publish_shared("root", "path", "old", {"path": _frame()})
    for values in ((3.0, 4.0), (5.0, 6.0)):
        with pytest.raises(RuntimeError):
            store.publish_shared("root", "path", "new", {"path": _frame(values)})
    plan = store.cleanup_plan()
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("cancel")

    with pytest.raises(InterruptedError):
        store.cleanup(plan, check_canceled=cancel)
    assert len(store.cleanup_plan()["candidates"]) == 1
    assert store.cleanup(plan)["deleted"] == 2
    assert store.cleanup_plan()["candidates"] == []


def test_unregistered_conflicting_file_cannot_be_published(tmp_path):
    from bagelquant_core.hashing import hash_dataframe

    store = _store(tmp_path)
    intended = _frame()
    path = store._table_path(hash_dataframe(intended))
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame((90.0, 91.0)).write_parquet(path)
    with pytest.raises(ValueError, match=r"unregistered.*differs"):
        store.publish_shared("root", "path", "new", {"path": intended})
    assert store.shared_receipt("root", "path") is None


def test_new_publication_indexes_only_new_table_in_index_free_store(tmp_path):
    store = _store(tmp_path)
    original = store.publish_shared("root", "path", "old", {"path": _frame()})
    with store._connect() as db:
        db.execute("DROP TABLE bt_table_index")
    store.initialize()
    assert (
        store.describe_reference(original["files"]["path"]) == original["files"]["path"]
    )
    updated = store.publish_shared(
        "root", "path", "new", {"path": _frame((3.0, 4.0))}, expected_shared=original
    )
    assert updated["files"]["path"] != original["files"]["path"]
    with store._read_connection() as db:
        assert db.execute("SELECT count(*) FROM bt_table_index").fetchone()[0] == 1


def test_empty_index_plan_is_passive_on_uninitialized_store(tmp_path):
    store = BTStore(tmp_path / "new" / "bt.sqlite", tmp_path / "new" / "tables")
    assert store.index_plan()["tables"] == []
    assert store.build_index(store.index_plan())["tables"] == 0
    assert not store.meta_path.parent.exists()


def test_index_maintenance_admission_returns_partial_before_full_decode(
    tmp_path, monkeypatch
):
    import bagelquant_core.resources as resources

    store = _store(tmp_path)
    store.publish_shared("root", "path", "old", {"path": _frame()})
    with store._connect() as db:
        db.execute("DELETE FROM bt_table_index")
    monkeypatch.setattr(resources, "admit_parquet_materialization", lambda _: False)
    monkeypatch.setattr(
        pl, "read_parquet", lambda *_a, **_k: pytest.fail("unadmitted table decoded")
    )
    result = store.build_index(store.index_plan())
    assert result["status"] == "partial"
    assert result["tables"] == 0 and result["unknown_tables"] == 1
