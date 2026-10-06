"""Public store readiness is read-only, including incompatible evidence."""

import sqlite3
from contextlib import closing

import pytest

from bagelquant_bt import BTStore


def test_inspect_missing_store_creates_nothing(tmp_path):
    store = BTStore(tmp_path / "missing" / "meta.sqlite", tmp_path / "artifacts")
    assert store.inspect()["status"] == "uninitialized"
    assert list(tmp_path.iterdir()) == []


def test_inspect_initialized_store_preserves_bytes(tmp_path):
    store = BTStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    before = store.meta_path.read_bytes()
    assert store.inspect() == {"status": "ready", "schema_version": 1, "reason": None}
    assert store.meta_path.read_bytes() == before
    assert list(store.artifact_path.iterdir()) == []


def test_inspect_old_schema_and_corrupt_store(tmp_path):
    path = tmp_path / "meta.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE old_evidence(value TEXT)")
    store = BTStore(path, tmp_path / "artifacts")
    before = path.read_bytes()
    assert store.inspect()["status"] == "incompatible"
    assert path.read_bytes() == before
    path.write_bytes(b"retained corrupt evidence")
    assert store.inspect()["reason"] == "metadata_unreadable"
    assert path.read_bytes() == b"retained corrupt evidence"
    assert not store.artifact_path.exists()


def test_matching_version_with_incomplete_columns_is_incompatible(tmp_path):
    store = BTStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with sqlite3.connect(store.meta_path) as connection:
        connection.execute(
            "ALTER TABLE bt_chapters RENAME COLUMN manifest_json TO invalid_column"
        )
    before = store.meta_path.read_bytes()
    assert store.inspect()["status"] == "incompatible"
    assert store.meta_path.read_bytes() == before


@pytest.mark.parametrize("version", [1, 2])
def test_checkpointed_wal_inspection_creates_no_sidecars(tmp_path, version):
    store = BTStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with closing(sqlite3.connect(store.meta_path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("UPDATE bt_store_schema SET version=?", (version,))
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert set(before) == {"meta.sqlite"}
    assert store.inspect()["status"] == ("ready" if version == 1 else "incompatible")
    assert {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    } == before


@pytest.mark.parametrize("base_version,version", [(1, 2), (2, 1)])
def test_active_wal_inspection_reads_committed_version_without_touching_index(
    tmp_path, base_version, version
):
    store = BTStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with closing(sqlite3.connect(store.meta_path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        version_before = version
        version = base_version
        connection.execute("UPDATE bt_store_schema SET version=?", (version,))
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        version = version_before
        connection.execute("UPDATE bt_store_schema SET version=?", (version,))
        connection.commit()
        before = {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        }
        assert store.inspect()["status"] == (
            "ready" if version == 1 else "incompatible"
        )
        assert {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        } == before


@pytest.mark.parametrize("committed_version", [1, 2])
def test_active_rollback_journal_never_confers_uncommitted_readiness(
    tmp_path, committed_version
):
    store = BTStore(tmp_path / "meta.sqlite", tmp_path / "artifacts")
    store.initialize()
    with closing(sqlite3.connect(store.meta_path)) as connection:
        connection.execute("CREATE TABLE pressure(payload BLOB)")
        version = committed_version
        connection.execute("UPDATE bt_store_schema SET version=?", (version,))
        connection.commit()
        connection.execute("PRAGMA cache_size=1")
        connection.execute("PRAGMA cache_spill=ON")
        connection.execute("BEGIN IMMEDIATE")
        version = 3 - committed_version
        connection.execute("UPDATE bt_store_schema SET version=?", (version,))
        connection.executemany("INSERT INTO pressure VALUES(?)", [(b"x" * 65536,)] * 40)
        journal = store.meta_path.with_name(store.meta_path.name + "-journal")
        assert any(journal.read_bytes()[:8])
        before = {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        }
        assert store.inspect()["reason"] == "metadata_unreadable"
        assert {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        } == before
        connection.rollback()
        assert store.inspect()["status"] == (
            "ready" if committed_version == 1 else "incompatible"
        )
