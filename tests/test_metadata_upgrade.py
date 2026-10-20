import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing, contextmanager
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt import BTStore
from bagelquant_bt.metadata_upgrade import apply, plan, rollback
from bagelquant_bt.store import _identity, _json, _table_evidence
from bagelquant_bt.table_composition import coordinates


def legacy(tmp_path, *, rows=6, long_names=False):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frame = pl.DataFrame(
        {
            "time": [date(2024, 1, 1) + timedelta(days=i) for i in range(rows)],
            "asset_id": [
                ("华" * 100 if long_names else "A") + str(i) for i in range(rows)
            ],
            "factor_return": [i / 1000 for i in range(rows)],
        }
    )
    evaluation = store.begin("example", settings={}, through=frame["time"].max())
    receipt = store.publish_chapter(evaluation, "summary", {"daily": frame})
    reference = receipt["files"]["daily"]
    prefix = store.publish_table(frame.head(rows // 2))
    suffix = store.publish_table(frame.slice(rows // 2))
    composed = store.compose_reference([{"reference": prefix}, {"reference": suffix}])
    oldframes = [
        (reference, frame),
        (prefix, frame.head(rows // 2)),
        (suffix, frame.slice(rows // 2)),
        (composed, frame),
    ]
    with store._connect() as connection:
        for ref, values in oldframes:
            payload = {
                "schema": {key: str(dtype) for key, dtype in values.schema.items()},
                "columns": values.columns,
                "coordinates": coordinates(values),
            }
            evidence = _table_evidence(values, axes=["time", "asset_id"])
            if evidence is not None:
                payload["evidence"] = evidence
            header = {
                **payload,
                "digest": _identity(
                    {
                        "version": "bt.table.index.v2",
                        "reference": ref,
                        **payload,
                    }
                ),
            }
            connection.execute(
                "UPDATE bt_table_index SET schema_json=? WHERE table_id=?",
                (_json(header), ref["table_id"]),
            )
        connection.execute("UPDATE bt_store_schema SET version=1")
    shutil.rmtree(store.artifact_path / "indexes")  # Synthetic fixture only.
    return store, frame, reference, composed, receipt


def receipt_bytes(meta):
    with closing(sqlite3.connect(meta)) as connection:
        return {
            name: connection.execute(f"SELECT * FROM {name} ORDER BY 1").fetchall()
            for name in (
                "bt_evaluations",
                "bt_chapters",
                "bt_shared",
                "bt_cleanup_receipts",
            )
        }


def no_numerical_reads(monkeypatch):
    def forbidden(*_a, **_k):
        pytest.fail("offline metadata conversion opened numerical Parquet")

    monkeypatch.setattr(pl, "read_parquet", forbidden)
    monkeypatch.setattr(pl, "scan_parquet", forbidden)
    monkeypatch.setattr(BTStore, "read_reference", forbidden)


def test_upgrade_preserves_receipts_references_values_evidence_and_rollback(
    tmp_path, monkeypatch
):
    store, frame, reference, composed, receipt = legacy(tmp_path)
    before = receipt_bytes(store.meta_path)
    files = {
        path.name: path.read_bytes()
        for path in (store.artifact_path / "tables").glob("*.parquet")
    }
    assert files
    assert plan(store.meta_path, store.artifact_path)["schema_version"] == 1
    with pytest.raises(ValueError, match="stop every"):
        apply(store.meta_path, store.artifact_path, tmp_path / "backups")
    with monkeypatch.context() as patch:
        no_numerical_reads(patch)
        result = apply(
            store.meta_path,
            store.artifact_path,
            tmp_path / "backups",
            confirm_stopped=True,
        )
    assert result["status"] == "complete"
    assert result["tables"] == 4
    assert receipt_bytes(store.meta_path) == before
    assert {
        path.name: path.read_bytes()
        for path in (store.artifact_path / "tables").glob("*.parquet")
    } == files
    assert store.chapter(receipt["evaluation_id"], "summary") == {
        **receipt,
        "status": "ready",
    }
    assert store.describe_reference(composed) == composed
    assert_frame_equal(store.read_reference(reference), frame)
    assert_frame_equal(store.read_reference(composed), frame)
    assert_frame_equal(
        store.read_reference(composed, coordinates=frame.select("time").tail(1)),
        frame.tail(1),
    )
    assert store.reference_evidence(reference) == store.reference_evidence(composed)
    store.verify_reference(reference, deep=True)
    store.verify_reference(composed, deep=True)
    assert (
        apply(
            store.meta_path,
            store.artifact_path,
            tmp_path / "backups",
            confirm_stopped=True,
        )["status"]
        == "already_converted"
    )
    assert (
        rollback(store.meta_path, store.artifact_path, confirm_stopped=True)["status"]
        == "rolled_back"
    )
    assert plan(store.meta_path, store.artifact_path)["schema_version"] == 1
    assert receipt_bytes(store.meta_path) == before
    with pytest.raises(ValueError, match="schema 2 required"):
        store.describe_reference(reference)


@pytest.mark.parametrize(
    "failure",
    [
        "backup_verified",
        "conversion_verified",
        "blocks_published",
        "database_replaced",
    ],
)
def test_interrupted_conversion_retries_without_losing_originals(
    tmp_path, failure, monkeypatch
):
    store, _, reference, _, _ = legacy(tmp_path)
    before = receipt_bytes(store.meta_path)
    no_numerical_reads(monkeypatch)

    def interrupt(stage):
        if stage == failure:
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        apply(
            store.meta_path,
            store.artifact_path,
            tmp_path / "backups",
            confirm_stopped=True,
            checkpoint=interrupt,
        )
    assert receipt_bytes(store.meta_path) == before
    result = apply(
        store.meta_path, store.artifact_path, tmp_path / "backups", confirm_stopped=True
    )
    assert result["status"] in {"complete", "already_converted"}
    assert store.describe_reference(reference) == reference
    assert len(list((tmp_path / "backups").glob("*/original.sqlite"))) == 1


def test_bad_legacy_digest_never_switches_database(tmp_path):
    store, _, _, _, _ = legacy(tmp_path)
    with closing(sqlite3.connect(store.meta_path)) as connection:
        connection.execute(
            "UPDATE bt_table_index SET "
            "schema_json=json_set(schema_json,'$.digest','wrong')"
        )
        connection.commit()
    before = receipt_bytes(store.meta_path)
    with pytest.raises(ValueError, match="digest differs"):
        apply(
            store.meta_path,
            store.artifact_path,
            tmp_path / "backups",
            confirm_stopped=True,
        )
    assert plan(store.meta_path, store.artifact_path)["schema_version"] == 1
    assert receipt_bytes(store.meta_path) == before


def test_rollback_rejects_new_writes_and_backup_corruption(tmp_path):
    store, _, _, _, _ = legacy(tmp_path)
    result = apply(
        store.meta_path, store.artifact_path, tmp_path / "backups", confirm_stopped=True
    )
    store.begin("new", settings={}, through="2024-01-09")
    with pytest.raises(ValueError, match="new writes"):
        rollback(store.meta_path, store.artifact_path, confirm_stopped=True)
    from pathlib import Path

    Path(result["backup"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="backup changed"):
        rollback(store.meta_path, store.artifact_path, confirm_stopped=True)


def test_large_json_cell_is_streamed_into_bounded_blocks(tmp_path, monkeypatch):
    # One SQLite cell exceeds the entire parser and decoded-block budget.
    store, _, _, _, _ = legacy(tmp_path, rows=28_000, long_names=True)
    assert (
        plan(store.meta_path, store.artifact_path)["largest_index_json_bytes"]
        > 8 * 1024 * 1024
    )
    no_numerical_reads(monkeypatch)
    result = apply(
        store.meta_path, store.artifact_path, tmp_path / "backups", confirm_stopped=True
    )
    assert result["peak_json_buffer_characters"] < 128 * 1024
    with closing(sqlite3.connect(store.meta_path)) as connection:
        for (value,) in connection.execute("SELECT schema_json FROM bt_table_index"):
            header = json.loads(value)
            assert len(value) < 20_000
            for descriptor in (
                header.get("coordinates"),
                (header.get("evidence") or {}).get("groups"),
            ):
                for block in (descriptor or {}).get("blocks", []):
                    assert block["decoded_bytes"] <= 8 * 1024 * 1024
                    assert block["rows"] <= 65_536


def test_committed_wal_is_included_and_active_writer_rejected(tmp_path):
    store, _, _, _, _ = legacy(tmp_path)
    connection = sqlite3.connect(store.meta_path)
    try:
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("UPDATE bt_evaluations SET owner_id='committed-wal'")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            apply(
                store.meta_path,
                store.artifact_path,
                tmp_path / "backups",
                confirm_stopped=True,
            )
        connection.rollback()
        # The other open WAL reader prevents the final DELETE-mode switch.
        connection.close()
        apply(
            store.meta_path,
            store.artifact_path,
            tmp_path / "backups",
            confirm_stopped=True,
        )
        assert receipt_bytes(store.meta_path)["bt_evaluations"][0][1] == "committed-wal"
    finally:
        connection.close()


def test_abrupt_process_exit_has_recorded_stage_and_retry_cleans_it(tmp_path):
    store, _, reference, _, _ = legacy(tmp_path)
    code = """
import os, sys
from bagelquant_bt.metadata_upgrade import apply
def crash(stage):
    if stage == 'conversion_verified':
        os._exit(71)
apply(*sys.argv[1:], confirm_stopped=True, checkpoint=crash)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(store.meta_path),
            str(store.artifact_path),
            str(tmp_path / "backups"),
        ],
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 71
    abandoned = list(tmp_path.glob(".bt-metadata-stage-*"))
    assert len(abandoned) == 1
    apply(
        store.meta_path, store.artifact_path, tmp_path / "backups", confirm_stopped=True
    )
    assert not list(tmp_path.glob(".bt-metadata-stage-*"))
    assert store.describe_reference(reference) == reference
    with store._read_connection() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.parametrize("after_install", [False, True])
def test_abrupt_rollback_exit_cleans_recorded_stage_on_retry(tmp_path, after_install):
    store, _, _, _, _ = legacy(tmp_path)
    before = receipt_bytes(store.meta_path)
    apply(
        store.meta_path, store.artifact_path, tmp_path / "backups", confirm_stopped=True
    )
    code = """
import os, sys
import bagelquant_bt.metadata_upgrade as upgrade
original = upgrade._install_database
def crash(*args, **kwargs):
    if sys.argv[3] == 'True':
        original(*args, **kwargs)
    os._exit(73)
upgrade._install_database = crash
upgrade.rollback(*sys.argv[1:3], confirm_stopped=True)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(store.meta_path),
            str(store.artifact_path),
            str(after_install),
        ],
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 73
    assert len(list(tmp_path.glob(".bt-metadata-stage-*"))) == 1
    result = rollback(store.meta_path, store.artifact_path, confirm_stopped=True)
    assert result["status"] == ("already_original" if after_install else "rolled_back")
    assert not list(tmp_path.glob(".bt-metadata-stage-*"))
    assert receipt_bytes(store.meta_path) == before
    assert plan(store.meta_path, store.artifact_path)["schema_version"] == 1


def test_reordered_fields_and_absent_none_capabilities_are_preserved(
    tmp_path, monkeypatch
):
    store, frame, reference, composed, _ = legacy(tmp_path)
    with closing(sqlite3.connect(store.meta_path)) as connection:
        for ref in (reference, composed):
            header = json.loads(
                connection.execute(
                    "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                    (ref["table_id"],),
                ).fetchone()[0]
            )
            header.pop("digest")
            if ref == reference:
                header.pop("columns")
                header.pop("evidence")
                header["coordinates"] = None
            else:
                header.pop("coordinates")
                header["evidence"] = None
            header["digest"] = _identity(
                {"version": "bt.table.index.v2", "reference": ref, **header}
            )
            connection.execute(
                "UPDATE bt_table_index SET schema_json=? WHERE table_id=?",
                (
                    json.dumps(
                        dict(reversed(list(header.items()))), ensure_ascii=False
                    ),
                    ref["table_id"],
                ),
            )
        connection.commit()
    no_numerical_reads(monkeypatch)
    apply(
        store.meta_path, store.artifact_path, tmp_path / "backups", confirm_stopped=True
    )
    with store._read_connection() as connection:
        first = json.loads(
            connection.execute(
                "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                (reference["table_id"],),
            ).fetchone()[0]
        )
        second = json.loads(
            connection.execute(
                "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                (composed["table_id"],),
            ).fetchone()[0]
        )
    assert (
        "columns" not in first
        and "evidence" not in first
        and first["coordinates"] is None
    )
    assert "coordinates" not in second and second["evidence"] is None
    assert store.reference_evidence(reference) is None
    assert frame.height == 6


def test_plan_preserves_copied_main_and_wal_without_creating_shm(tmp_path):
    store, _, _, _, _ = legacy(tmp_path)
    with closing(sqlite3.connect(store.meta_path)) as writer:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("UPDATE bt_evaluations SET owner_id='committed'")
        writer.commit()
        copied = tmp_path / "copied.sqlite"
        shutil.copyfile(store.meta_path, copied)
        shutil.copyfile(str(store.meta_path) + "-wal", str(copied) + "-wal")
        before = {
            path.name: path.read_bytes() for path in tmp_path.glob("copied.sqlite*")
        }
        assert plan(copied, store.artifact_path)["schema_version"] == 1
        assert {
            path.name: path.read_bytes() for path in tmp_path.glob("copied.sqlite*")
        } == before


def test_empty_coordinate_and_evidence_arrays_remain_available(tmp_path, monkeypatch):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frame = pl.DataFrame(
        schema={"time": pl.Date, "asset_id": pl.String, "factor_return": pl.Float64}
    )
    reference = store.publish_table(frame)
    payload = {
        "schema": {key: str(dtype) for key, dtype in frame.schema.items()},
        "columns": frame.columns,
        "coordinates": [],
        "evidence": _table_evidence(frame, axes=["time", "asset_id"]),
    }
    assert payload["evidence"]["groups"] == []
    payload["digest"] = _identity(
        {"version": "bt.table.index.v2", "reference": reference, **payload}
    )
    with store._connect() as connection:
        connection.execute(
            "UPDATE bt_table_index SET schema_json=? WHERE table_id=?",
            (_json(payload), reference["table_id"]),
        )
        connection.execute("UPDATE bt_store_schema SET version=1")
    with monkeypatch.context() as patch:
        no_numerical_reads(patch)
        apply(
            store.meta_path,
            store.artifact_path,
            tmp_path / "backups",
            confirm_stopped=True,
        )
    with store._read_connection() as connection:
        header = json.loads(
            connection.execute("SELECT schema_json FROM bt_table_index").fetchone()[0]
        )
    assert header["coordinates"]["blocks"] == []
    assert header["evidence"]["groups"]["blocks"] == []
    assert store.reference_evidence(reference) is not None
    assert_frame_equal(store.read_reference(reference), frame)
    store.verify_reference(reference, deep=True)


def test_rollback_rejects_partial_converted_backup(tmp_path):
    from pathlib import Path

    store, _, _, _, _ = legacy(tmp_path)
    result = apply(
        store.meta_path, store.artifact_path, tmp_path / "backups", confirm_stopped=True
    )
    Path(result["backup"]).with_name("converted.sqlite").write_bytes(b"partial")
    with pytest.raises(sqlite3.DatabaseError):
        rollback(store.meta_path, store.artifact_path, confirm_stopped=True)
    assert store.inspect()["status"] == "ready"


def test_new_directory_entries_are_durable_before_database_replacement(
    tmp_path, monkeypatch
):
    import bagelquant_bt.metadata_upgrade as upgrade

    store, _, _, _, _ = legacy(tmp_path)
    backup_root = tmp_path / "new" / "nested" / "backups"
    synced = []
    original_sync = upgrade._sync_directory
    original_install = upgrade._install_database

    def record_sync(path):
        synced.append(path)
        original_sync(path)

    def checked_install(*args, **kwargs):
        # Ancestors of both preservation and published-block directories must
        # reach durable storage before the new database can reference them.
        for directory in (
            tmp_path,
            tmp_path / "new",
            backup_root.parent,
            backup_root,
            store.artifact_path,
            store.artifact_path / "indexes",
        ):
            assert directory in synced
        return original_install(*args, **kwargs)

    monkeypatch.setattr(upgrade, "_sync_directory", record_sync)
    monkeypatch.setattr(upgrade, "_install_database", checked_install)
    apply(store.meta_path, store.artifact_path, backup_root, confirm_stopped=True)


def test_plan_sizes_blob_headers_without_reading_json_cells(tmp_path, monkeypatch):
    import bagelquant_bt.metadata_upgrade as upgrade

    store, _, _, _, _ = legacy(tmp_path, long_names=True)
    with closing(sqlite3.connect(store.meta_path)) as connection:
        sizes = [
            len(value.encode("utf-8"))
            for (value,) in connection.execute("SELECT schema_json FROM bt_table_index")
        ]
    original_snapshot = upgrade.open_metadata_snapshot
    observed = []

    class LengthOnly:
        def __init__(self, blob):
            self.blob = blob

        def __len__(self):
            return len(self.blob)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.blob.close()

        def read(self, *_):
            pytest.fail("planning read JSON body bytes")

    class HeaderConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, *args):
            assert "schema_json" not in sql
            return self.connection.execute(sql, *args)

        def blobopen(self, *args, **kwargs):
            observed.append(args[2])
            return LengthOnly(self.connection.blobopen(*args, **kwargs))

    @contextmanager
    def snapshot(*args, **kwargs):
        with original_snapshot(*args, **kwargs) as connection:
            yield HeaderConnection(connection)

    monkeypatch.setattr(upgrade, "open_metadata_snapshot", snapshot)
    result = plan(store.meta_path, store.artifact_path)
    assert len(observed) == result["index_rows"] == 4
    assert result["index_json_bytes"] == sum(sizes)
    assert result["largest_index_json_bytes"] == max(sizes)


def test_directory_retry_syncs_existing_entry_and_parent(tmp_path, monkeypatch):
    import bagelquant_bt.metadata_upgrade as upgrade

    path = tmp_path / "created-before-crash"
    path.mkdir()
    synced = []
    monkeypatch.setattr(upgrade, "_sync_directory", synced.append)
    upgrade._mkdir_durable(path)
    assert synced == [path, path.parent]
