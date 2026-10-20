"""Explicit offline, lossless schema-1 to schema-2 metadata conversion.

Nothing in normal BTStore startup imports or invokes this module. Call ``plan``
first; ``apply`` and ``rollback`` require all users of this store to be stopped.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import closing, contextmanager
from datetime import date
from pathlib import Path

import polars as pl
from bagelquant_core.inspection import open_metadata_snapshot

from .index_blocks import MAX_BYTES, MAX_ROWS, MetadataBlocks
from .legacy_index import read_legacy_index
from .store import BTStore, _file_hash, _json
from .table_composition import COORDINATES

TABLES = (
    "bt_store_schema",
    "bt_evaluations",
    "bt_chapters",
    "bt_shared",
    "bt_cleanup_receipts",
    "bt_table_index",
)


@contextmanager
def _open(path, *, readonly=True):
    with closing(
        sqlite3.connect(
            Path(path).as_uri() + ("?mode=ro" if readonly else "?mode=rw"),
            uri=True,
            timeout=0,
        )
    ) as connection:
        connection.execute("PRAGMA cache_size=-4096")
        if readonly:
            connection.execute("PRAGMA query_only=ON")
        yield connection


def _version(connection):
    versions = connection.execute("SELECT version FROM bt_store_schema").fetchall()
    if len(versions) != 1 or versions[0][0] not in (1, 2):
        raise ValueError("offline conversion accepts BT metadata schema 1 or 2 only")
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE "
            "'sqlite_%'"
        )
    }
    if tables not in (set(TABLES), set(TABLES) - {"bt_table_index"}):
        raise ValueError("unsupported BT tables; inspect the store before conversion")
    return versions[0][0], tables


def _fingerprint(connection):
    """Hash committed logical bytes, independent of WAL/checkpoint page layout."""
    _, tables = _version(connection)
    digest = hashlib.sha256()
    for table in sorted(tables):
        digest.update(table.encode() + b"\0")
        if table == "bt_table_index":
            for rowid, identity, reference in connection.execute(
                "SELECT rowid,table_id,reference_json FROM bt_table_index ORDER BY "
                "table_id"
            ):
                digest.update(_json([identity, reference]).encode())
                with connection.blobopen(
                    table, "schema_json", rowid, readonly=True
                ) as blob:
                    digest.update(str(len(blob)).encode() + b":")
                    for chunk in iter(lambda: blob.read(64 * 1024), b""):
                        digest.update(chunk)
        else:
            columns = [
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            ]
            order = ",".join('"' + name + '"' for name in columns)
            for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                digest.update(_json(row).encode() + b"\n")
    return digest.hexdigest()


def plan(meta_path, artifact_path):
    """Inspect counts and conservative disk needs; never initialize or migrate."""
    meta, artifacts = _paths(meta_path, artifact_path)
    with open_metadata_snapshot(meta, runtime=False) as connection:
        connection.execute("BEGIN")
        version, tables = _version(connection)
        count, total, largest = (0, 0, 0)
        if "bt_table_index" in tables:
            # SQLite length(CAST(TEXT AS BLOB)) materializes the entire cell.
            # Blob lengths use only the record header, even for giant JSON cells.
            for (rowid,) in connection.execute("SELECT rowid FROM bt_table_index"):
                with connection.blobopen(
                    "bt_table_index", "schema_json", rowid, readonly=True
                ) as blob:
                    size = len(blob)
                count += 1
                total += size
                largest = max(largest, size)
        size = (
            connection.execute("PRAGMA page_count").fetchone()[0]
            * connection.execute("PRAGMA page_size").fetchone()[0]
        )
    return {
        "schema_version": version,
        "meta_path": str(meta),
        "artifact_path": str(artifacts),
        "index_rows": count,
        "index_json_bytes": total,
        "largest_index_json_bytes": largest,
        "database_bytes": size,
        "maximum_block_rows": MAX_ROWS,
        "maximum_decoded_block_bytes": MAX_BYTES,
        # Backup + stage + blocks + temporary equivalence inventory. This is an
        # admission estimate, not a promised compression ratio.
        "recommended_free_bytes": size * 4,
        "requires_stopped_users": True,
        "numerical_files_read": 0,
    }


def _paths(meta_path, artifact_path):
    meta, artifacts = Path(meta_path).absolute(), Path(artifact_path).absolute()
    if meta.is_symlink() or artifacts.is_symlink() or not meta.is_file():
        raise ValueError(
            "conversion requires an existing regular metadata file and real artifact "
            "root"
        )
    return meta.resolve(), artifacts.resolve()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path):
    """Persist every new directory entry before publishing recovery authority."""
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    # A retry may find a directory created just before a previous process died.
    _sync_directory(cursor)
    _sync_directory(cursor.parent)
    for directory in reversed(missing):
        directory.mkdir()
        _sync_directory(directory)
        _sync_directory(directory.parent)
    if not path.is_dir():
        raise ValueError(f"conversion directory is not a directory: {path}")


def _write_state(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _locked(meta, confirm_stopped):
    if not confirm_stopped:
        raise ValueError(
            "stop every reader/writer, then explicitly pass confirm_stopped=True"
        )
    lock = meta.with_name(meta.name + ".upgrade.lock")
    if lock.is_symlink():
        raise ValueError("conversion lock cannot be a symlink")
    with lock.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another BT metadata conversion holds the lock") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _batch_schema(schema, columns, *, evidence=False):
    result = {}
    for name in columns:
        dtype = schema.get(name)
        if dtype not in {"Date", "String", "Boolean", "Null"} and not (
            isinstance(dtype, str) and dtype.startswith(("Int", "UInt", "Float"))
        ):
            raise ValueError(
                f"unsupported legacy BT metadata coordinate type: {name}={dtype}"
            )
        result[name] = getattr(pl, dtype)
    result["_rows"] = pl.UInt64
    if evidence:
        result.update(_a=pl.UInt64, _b=pl.UInt64)
    return result


def _batches(records, schema, *, evidence=False):
    batch, size = [], 0
    names = list(schema)
    for record in records:
        encoded_size = len(_json(record).encode())
        if encoded_size > MAX_BYTES:
            raise ValueError("legacy BT metadata row exceeds the block budget")
        if batch and (len(batch) >= MAX_ROWS or size + encoded_size > MAX_BYTES):
            yield pl.DataFrame(batch, schema=schema)
            batch, size = [], 0
        if evidence:
            if not isinstance(record, list) or len(record) != len(names):
                raise ValueError("invalid legacy BT evidence group")
            record = dict(zip(names, record, strict=True))
            record["_a"], record["_b"] = int(record["_a"], 16), int(record["_b"], 16)
        elif set(record) != set(names):
            raise ValueError("legacy BT coordinate columns differ")
        if type(record["_rows"]) is not int or record["_rows"] < 0:
            raise ValueError("invalid legacy BT coordinate row count")
        converted = {
            key: date.fromisoformat(value)
            if schema[key] == pl.Date and value is not None
            else value
            for key, value in record.items()
        }
        batch.append(converted)
        size += encoded_size
    if batch:
        yield pl.DataFrame(batch, schema=schema)


def _array_digest(metadata, descriptor, *, evidence=False):
    digest, count = hashlib.sha256(b"["), 0
    for frame in metadata.frames(descriptor):
        for record in frame.iter_rows(named=not evidence):
            if evidence:
                record = [
                    *record[:-2],
                    format(record[-2], "016x"),
                    format(record[-1], "016x"),
                ]
            digest.update((b"," if count else b"") + _json(record).encode())
            count += 1
    digest.update(b"]")
    return digest.hexdigest(), count


def _prove_composition(connection, metadata, reference, descriptor):
    """Compare the exact legacy coordinate multiset to filtered leaf metadata.

    This disposable, disk-backed equality check has a 4 MiB SQLite cache. It is
    not saved as BT authority and never opens a numerical file.
    """
    with tempfile.TemporaryDirectory(
        prefix="coordinate-proof-", dir=metadata.root.parent.parent
    ) as directory:
        with closing(sqlite3.connect(Path(directory) / "proof.sqlite")) as proof:
            proof.execute("PRAGMA cache_size=-4096")
            proof.execute("PRAGMA journal_mode=OFF")
            proof.execute(
                "CREATE TABLE counts(key TEXT PRIMARY KEY, n INTEGER NOT NULL)"
            )
            for frame in metadata.frames(descriptor):
                proof.executemany(
                    "INSERT INTO counts VALUES(?,?)",
                    (
                        (
                            _json(
                                {
                                    key: value
                                    for key, value in row.items()
                                    if key != "_rows"
                                }
                            ),
                            row["_rows"],
                        )
                        for row in frame.iter_rows(named=True)
                    ),
                )
            for part in reference["parts"]:
                row = connection.execute(
                    "SELECT reference_json,schema_json FROM bt_table_index WHERE "
                    "table_id=?",
                    (part["reference"]["table_id"],),
                ).fetchone()
                if row is None or json.loads(row[0]) != part["reference"]:
                    raise ValueError(
                        "legacy composition has an unavailable original reference"
                    )
                index = json.loads(row[1])
                if (
                    not isinstance(index.get("coordinates"), dict)
                    or "blocks" not in index["coordinates"]
                ):
                    raise ValueError(
                        "legacy composition needs original coordinate-index maintenance"
                    )
                for frame in metadata.frames(
                    index["coordinates"], predicates=part["filters"]
                ):
                    proof.executemany(
                        "INSERT INTO counts VALUES(?,?) ON CONFLICT(key) DO UPDATE "
                        "SET n=n+excluded.n",
                        (
                            (
                                _json(
                                    {
                                        key: value
                                        for key, value in row.items()
                                        if key != "_rows"
                                    }
                                ),
                                -row["_rows"],
                            )
                            for row in frame.iter_rows(named=True)
                        ),
                    )
            if proof.execute("SELECT 1 FROM counts WHERE n != 0 LIMIT 1").fetchone():
                raise ValueError(
                    "legacy composed coordinates differ from original parts"
                )


def _convert(source, target, artifacts):
    """Copy receipts exactly; stream and independently round-trip each index."""
    metadata = MetadataBlocks(artifacts)
    peak, converted = 0, 0
    with _open(source) as old, closing(sqlite3.connect(target)) as new:
        version, tables = _version(old)
        if version != 1:
            raise ValueError("conversion source must be schema 1")
        old.execute("BEGIN")
        new.execute("PRAGMA cache_size=-4096")
        for (statement,) in old.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE "
            "'sqlite_%' ORDER BY name"
        ):
            new.execute(statement)
        for table in sorted(tables - {"bt_table_index"}):
            columns = old.execute(f"PRAGMA table_info({table})").fetchall()
            placeholders = ",".join("?" for _ in columns)
            new.executemany(
                f"INSERT INTO {table} VALUES({placeholders})",
                old.execute(f"SELECT * FROM {table}"),
            )
        if "bt_table_index" in tables:
            # Originals precede compositions; public references contain only
            # original leaves. Never SELECT the giant schema_json cell.
            for rowid, identity, reference_json in old.execute(
                "SELECT rowid,table_id,reference_json FROM bt_table_index "
                "ORDER BY json_extract(reference_json,'$.format')='concat',table_id"
            ):
                reference = json.loads(reference_json)
                if reference["table_id"] != identity:
                    raise ValueError("legacy BT reference identity differs")
                header, arrays, used = read_legacy_index(old, rowid, reference)
                peak = max(peak, used)
                columns = [
                    key
                    for key in COORDINATES
                    if key in header["schema"]
                    and (
                        not arrays.get("coordinates", {}).get("columns")
                        or key in arrays["coordinates"]["columns"]
                    )
                ]
                coordinate_schema = _batch_schema(header["schema"], columns)
                evidence_schema = (
                    _batch_schema(
                        header["schema"], header["evidence"]["axes"], evidence=True
                    )
                    if header.get("evidence") is not None
                    else None
                )
                compact, repeated, used = read_legacy_index(
                    old,
                    rowid,
                    reference,
                    coordinates=lambda records, schema=coordinate_schema: (
                        metadata.save_batches(_batches(records, schema), schema=schema)
                    ),
                    groups=lambda records, schema=evidence_schema: (
                        metadata.save_batches(
                            _batches(records, schema, evidence=True), schema=schema
                        )
                    ),
                )
                peak = max(peak, used)
                if arrays != repeated:
                    raise ValueError("legacy index changed during conversion")
                for name, facts in arrays.items():
                    descriptor = (
                        compact["coordinates"]
                        if name == "coordinates"
                        else compact["evidence"]["groups"]
                    )
                    actual = _array_digest(
                        metadata, descriptor, evidence=name == "groups"
                    )
                    if actual != (facts["digest"], facts["rows"]):
                        raise ValueError("converted BT metadata round trip differs")
                if reference.get("format") == "concat" and "coordinates" in arrays:
                    _prove_composition(new, metadata, reference, compact["coordinates"])
                    compact["coordinates"] = {"composition": True}
                new.execute(
                    "INSERT INTO bt_table_index VALUES(?,?,?)",
                    (identity, reference_json, BTStore._seal_index(reference, compact)),
                )
                converted += 1
        for (statement,) in old.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL "
            "ORDER BY name"
        ):
            new.execute(statement)
        new.execute("UPDATE bt_store_schema SET version=2")
        new.commit()
        if new.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("converted BT database failed integrity_check")
        fingerprint = _fingerprint(new)
        retained = set()
        if "bt_table_index" in tables:
            for (payload,) in new.execute("SELECT schema_json FROM bt_table_index"):
                retained.update(metadata.identifiers(json.loads(payload)))
        new.execute("PRAGMA journal_mode=WAL")
    # Drop only temporary composed-coordinate chunks that no descriptor uses.
    if metadata.root.exists():
        for path in metadata.root.iterdir():
            if path.stem not in retained:
                path.unlink()
    return {
        "tables": converted,
        "peak_json_buffer_characters": peak,
        "fingerprint": fingerprint,
    }


def _install_database(meta, stage, *, expected):
    """Checkpoint the stopped source and replace its main DB as the final write."""
    with _open(meta, readonly=False) as connection:
        if _fingerprint(connection) != expected:
            raise ValueError("BT source changed; refusing database replacement")
        if connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
            raise ValueError("BT WAL is busy; stop all users before conversion")
        if connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
            raise ValueError("BT readers are still active; conversion remains staged")
        connection.execute("BEGIN EXCLUSIVE")
        if _fingerprint(connection) != expected:
            raise ValueError("BT source changed before final publication")
        with stage.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(stage, meta)
        _sync_directory(meta.parent)
        connection.rollback()


def _discard_stage(meta, state):
    """Remove only this journal's marked, abandoned staging directory."""
    if state.get("stage") is None:
        return
    stage = Path(state["stage"])
    if not stage.exists():
        return
    marker = stage / "owner.json"
    if (
        stage.is_symlink()
        or stage.parent != meta.parent
        or not stage.name.startswith(".bt-metadata-stage-")
        or marker.is_symlink()
        or not marker.is_file()
        or json.loads(marker.read_text())
        != {
            "meta_path": str(meta),
            "backup": state["backup"],
        }
    ):
        raise ValueError("unverified conversion stage; inspect it before recovery")
    shutil.rmtree(stage)


def apply(
    meta_path,
    artifact_path,
    backup_directory,
    *,
    confirm_stopped=False,
    checkpoint=lambda _: None,
):
    """Back up, verify, convert and publish; reruns recover interrupted attempts.

    ``checkpoint`` is an instrumentation hook for crash/retry acceptance tests.
    No numerical Parquet read or financial calculation is performed.
    """
    meta, artifacts = _paths(meta_path, artifact_path)
    journal = meta.with_name(meta.name + ".upgrade.json")
    with _locked(meta, confirm_stopped):
        if journal.is_symlink():
            raise ValueError("conversion journal cannot be a symlink")
        state = json.loads(journal.read_text()) if journal.exists() else None
        with _open(meta) as source:
            version, _ = _version(source)
            current = _fingerprint(source)
        if state is not None and (
            state["meta_path"] != str(meta) or state["artifact_path"] != str(artifacts)
        ):
            raise ValueError("conversion journal belongs to another store")
        if version == 2:
            if state is not None and current == state.get("converted_fingerprint"):
                _verify_published(meta, artifacts)
                _discard_stage(meta, state)
                state["status"] = "complete"
                _write_state(journal, state)
            return {
                "status": "already_converted",
                "journal": str(journal) if state else None,
            }
        if state is None:
            backup_root = Path(backup_directory).resolve()
            _mkdir_durable(backup_root)
            info = plan(meta, artifacts)
            # All destinations must have room independently; cross-device
            # publication copies blocks before the database switch.
            for directory in (backup_root, meta.parent, artifacts.parent):
                if shutil.disk_usage(directory).free < info["recommended_free_bytes"]:
                    raise ValueError(
                        f"insufficient conversion staging space in {directory}"
                    )
            bundle = backup_root / ("bt-metadata-" + uuid.uuid4().hex)
            _mkdir_durable(bundle)
            state = {
                "version": 1,
                "status": "started",
                "meta_path": str(meta),
                "artifact_path": str(artifacts),
                "backup": str(bundle / "original.sqlite"),
                "source_fingerprint": current,
            }
            _write_state(journal, state)
        if current != state["source_fingerprint"]:
            raise ValueError("BT source changed since this conversion began")
        _discard_stage(meta, state)
        backup = Path(state["backup"])
        with _open(meta, readonly=False) as writer:
            writer.execute(
                "BEGIN IMMEDIATE"
            )  # Exclude writers through backup/conversion.
            if _fingerprint(writer) != state["source_fingerprint"]:
                raise ValueError("BT source changed before backup")
            if state.get("backup_sha256") is None:
                # A failed partial backup is never accepted as preservation evidence.
                pending = backup.with_suffix(".partial")
                pending.unlink(missing_ok=True)
                with (
                    _open(meta) as source,
                    closing(sqlite3.connect(pending)) as destination,
                ):
                    source.backup(destination, pages=256)
                    if destination.execute("PRAGMA integrity_check").fetchall() != [
                        ("ok",)
                    ]:
                        raise ValueError("BT backup failed integrity_check")
                    if _fingerprint(destination) != state["source_fingerprint"]:
                        raise ValueError("BT backup differs from committed source")
                os.replace(pending, backup)
                with backup.open("rb") as stream:
                    os.fsync(stream.fileno())
                _sync_directory(backup.parent)
                state["backup_sha256"] = _file_hash(backup)
                state["status"] = "backed_up"
                _write_state(journal, state)
            if _file_hash(backup) != state["backup_sha256"]:
                raise ValueError("BT conversion backup changed; refusing conversion")
            checkpoint("backup_verified")
            with tempfile.TemporaryDirectory(
                prefix=".bt-metadata-stage-", dir=meta.parent
            ) as temporary:
                stage = Path(temporary)
                _sync_directory(stage.parent)
                _write_state(
                    stage / "owner.json",
                    {
                        "meta_path": str(meta),
                        "backup": state["backup"],
                    },
                )
                state["stage"] = str(stage)
                _write_state(journal, state)
                converted = stage / "converted.sqlite"
                result = _convert(backup, converted, stage / "artifacts")
                state.update(
                    converted_fingerprint=result["fingerprint"],
                    conversion=result,
                    status="converted",
                )
                _write_state(journal, state)
                checkpoint("conversion_verified")
                source_blocks = MetadataBlocks(stage / "artifacts")
                target_blocks = MetadataBlocks(artifacts)
                if source_blocks.root.exists():
                    _mkdir_durable(target_blocks.root)
                    for path in sorted(source_blocks.root.iterdir()):
                        destination = target_blocks.path(path.stem)
                        if destination.exists():
                            if _file_hash(destination) != path.stem:
                                raise ValueError(
                                    "conflicting published BT metadata block"
                                )
                        else:
                            pending = destination.with_suffix(
                                "." + uuid.uuid4().hex + ".tmp"
                            )
                            try:
                                shutil.copyfile(path, pending)
                                with pending.open("rb") as stream:
                                    os.fsync(stream.fileno())
                                if _file_hash(pending) != path.stem:
                                    raise ValueError("copied BT metadata block differs")
                                os.link(pending, destination)
                            finally:
                                pending.unlink(missing_ok=True)
                    _sync_directory(target_blocks.root)
                checkpoint("blocks_published")
                writer.rollback()
                writer.close()
                _install_database(meta, converted, expected=state["source_fingerprint"])
                checkpoint("database_replaced")
                _verify_published(meta, artifacts)
                state["status"] = "complete"
                _write_state(journal, state)
        return {
            "status": "complete",
            "backup": str(backup),
            "journal": str(journal),
            **result,
        }


def _verify_published(meta, artifacts):
    metadata = MetadataBlocks(artifacts)
    with _open(meta) as connection:
        version, tables = _version(connection)
        if version != 2:
            raise ValueError("converted BT schema is unavailable")
        if "bt_table_index" not in tables:
            return
        for reference_json, header in connection.execute(
            "SELECT reference_json,schema_json FROM bt_table_index"
        ):
            index = json.loads(header)
            payload = {key: value for key, value in index.items() if key != "digest"}
            if BTStore._seal_index(json.loads(reference_json), payload) != header:
                raise ValueError("converted BT index header differs")
            descriptors = (
                index.get("coordinates"),
                (index.get("evidence") or {}).get("groups"),
            )
            for descriptor in descriptors:
                if isinstance(descriptor, dict) and "blocks" in descriptor:
                    for _ in metadata.frames(descriptor):
                        pass


def rollback(meta_path, artifact_path, *, confirm_stopped=False):
    """Restore the verified schema-1 backup only while converted data is unchanged."""
    meta, artifacts = _paths(meta_path, artifact_path)
    journal = meta.with_name(meta.name + ".upgrade.json")
    with _locked(meta, confirm_stopped):
        if journal.is_symlink():
            raise ValueError("conversion journal cannot be a symlink")
        state = json.loads(journal.read_text())
        if state["meta_path"] != str(meta) or state["artifact_path"] != str(artifacts):
            raise ValueError("conversion journal belongs to another store")
        backup = Path(state["backup"])
        if _file_hash(backup) != state["backup_sha256"]:
            raise ValueError("BT conversion backup changed; refusing rollback")
        with _open(meta) as connection:
            current = _fingerprint(connection)
        if current == state["source_fingerprint"]:
            _discard_stage(meta, state)
            state["status"] = "rolled_back"
            _write_state(journal, state)
            return {"status": "already_original", "backup": str(backup)}
        if current != state.get("converted_fingerprint"):
            raise ValueError(
                "converted BT store has new writes; rollback would lose them"
            )
        _discard_stage(meta, state)
        with tempfile.TemporaryDirectory(
            prefix=".bt-metadata-stage-", dir=meta.parent
        ) as directory:
            stage = Path(directory)
            _sync_directory(stage.parent)
            _write_state(
                stage / "owner.json",
                {"meta_path": str(meta), "backup": state["backup"]},
            )
            state["stage"] = str(stage)
            state["status"] = "rolling_back"
            _write_state(journal, state)
            restored = stage / "restored.sqlite"
            shutil.copyfile(backup, restored)
            # Retain the converted database as recovery evidence too.
            saved = backup.with_name("converted.sqlite")
            if not saved.exists():
                pending = saved.with_suffix(".partial")
                pending.unlink(missing_ok=True)
                with (
                    _open(meta) as source,
                    closing(sqlite3.connect(pending)) as destination,
                ):
                    source.backup(destination, pages=256)
                    if destination.execute("PRAGMA integrity_check").fetchall() != [
                        ("ok",)
                    ]:
                        raise ValueError(
                            "converted-state backup failed integrity_check"
                        )
                    if _fingerprint(destination) != current:
                        raise ValueError("converted-state backup differs")
                with pending.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(pending, saved)
                _sync_directory(saved.parent)
            with _open(saved) as preserved:
                if (
                    preserved.execute("PRAGMA integrity_check").fetchall() != [("ok",)]
                    or _fingerprint(preserved) != current
                ):
                    raise ValueError("retained converted-state backup differs")
            _install_database(meta, restored, expected=current)
        state["status"] = "rolled_back"
        _write_state(journal, state)
        return {
            "status": "rolled_back",
            "backup": str(backup),
            "retained_blocks": str(artifacts / "indexes"),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "apply", "rollback"))
    parser.add_argument("--meta-path", type=Path, required=True)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--backup-directory", type=Path)
    parser.add_argument("--confirm-stopped", action="store_true")
    args = parser.parse_args()
    if args.action == "plan":
        result = plan(args.meta_path, args.artifact_path)
    elif args.action == "apply":
        if args.backup_directory is None:
            parser.error("apply requires --backup-directory")
        result = apply(
            args.meta_path,
            args.artifact_path,
            args.backup_directory,
            confirm_stopped=args.confirm_stopped,
        )
    else:
        result = rollback(
            args.meta_path, args.artifact_path, confirm_stopped=args.confirm_stopped
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
