"""BT-owned immutable evaluation tables, publication and recovery evidence."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import copy
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from bagelquant_core import ArtifactVerification
from bagelquant_core.hashing import hash_dataframe
from bagelquant_core.inspection import open_metadata_snapshot

_SCHEMA = """
CREATE TABLE bt_store_schema(version INTEGER NOT NULL);
INSERT INTO bt_store_schema VALUES(1);
CREATE TABLE bt_evaluations(
 id TEXT PRIMARY KEY,owner_id TEXT NOT NULL,manifest_json TEXT NOT NULL,
 created_at TEXT NOT NULL);
CREATE INDEX bt_evaluation_owner ON bt_evaluations(owner_id,created_at);
CREATE TABLE bt_chapters(
 evaluation_id TEXT NOT NULL,section TEXT NOT NULL,manifest_json TEXT NOT NULL,
 invalid_reason TEXT,created_at TEXT NOT NULL,
 PRIMARY KEY(evaluation_id,section));
CREATE TABLE bt_shared(
 owner_id TEXT NOT NULL,component TEXT NOT NULL,input_identity TEXT NOT NULL,
 receipt_identity TEXT NOT NULL,manifest_json TEXT NOT NULL,created_at TEXT NOT NULL,
 PRIMARY KEY(owner_id,component,input_identity,receipt_identity));
CREATE TABLE bt_cleanup_receipts(
 id TEXT PRIMARY KEY,manifest_json TEXT NOT NULL,completed_json TEXT NOT NULL);
"""


def _json(value: Any) -> str:
    def encode(item):
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        raise TypeError(f"BT metadata requires JSON values, got {type(item).__name__}")

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=encode
    )


def _identity(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _path(path: Path) -> str:
    return str(path)


def _file_hash(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with open(_path(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("BT artifact changed during verification")
    return digest.hexdigest()


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _verify_record(record: Mapping[str, Any], schema: str, **keys) -> None:
    """Validate the immutable payload and its stored SQLite coordinates."""
    if not isinstance(record, Mapping):
        raise ValueError("BT receipt must be an object")
    if schema not in {"bt.evaluation.v1", "bt.chapter.v1", "bt.shared.v1"}:
        raise ValueError("unknown BT receipt schema")
    if record.get("schema") != schema:
        raise ValueError("BT receipt schema differs")
    excluded = (
        {"id", "created_at"} if schema == "bt.evaluation.v1" else {"receipt_identity"}
    )
    if schema == "bt.chapter.v1":
        excluded.update({"updated_at", "status"})
    identity_name = "id" if schema == "bt.evaluation.v1" else "receipt_identity"
    payload = {name: value for name, value in record.items() if name not in excluded}
    if _identity(payload) != record.get(identity_name):
        raise ValueError("BT receipt payload identity differs")
    if any(record.get(name) != value for name, value in keys.items()):
        raise ValueError("BT receipt storage coordinates differ")


def _decode_record(row, schema, **keys):
    try:
        record = json.loads(row["manifest_json"])
        _verify_record(record, schema, **keys)
        return record
    except (TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("BT receipt metadata is malformed") from error


def evaluation_identity(
    mode: str,
    *,
    inputs: Mapping[str, Any],
    settings: Mapping[str, Any],
    kernel: str,
) -> str:
    """Hash explicit numerical inputs and policy, independent of row scan order.

    Frames use canonical column and row ordering. Opaque backend evidence refs
    may accompany them; this function neither resolves nor verifies those refs.
    Runtime resource ceilings are supplied separately, never as policy settings.
    """
    if mode not in {"signal", "weights", "execution", "input_evidence"}:
        raise ValueError("unsupported evaluation identity mode")
    _identifier(kernel, "kernel")

    return _identity(
        {
            "schema": "bt.input.identity.v1",
            "mode": mode,
            "kernel": kernel,
            "inputs": _normalize_inputs(inputs),
            "settings": _normalize_inputs(settings),
        }
    )


def _normalize_inputs(value):
    if isinstance(value, pl.DataFrame):
        selected = value.select(sorted(value.columns))
        return {"table": hash_dataframe(selected.sort(selected.columns))}
    if isinstance(value, Mapping):
        return {str(name): _normalize_inputs(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_inputs(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


class BTStore:
    """An explicit SQLite authority with immutable, content-addressed Parquet.

    Caller metadata and source receipt references are opaque. No input discovery,
    source verification, governance, numerical execution or machine policy occurs
    here. Table references disclose no path and are read only through this API.
    """

    def __init__(self, meta_path: str | Path, artifact_path: str | Path):
        self.meta_path = Path(meta_path).resolve()
        self.artifact_path = Path(artifact_path).resolve()
        self._verification: ArtifactVerification | None = None

    @contextmanager
    def read_context(self):
        """Share checksum proofs only for this finite operation, across threads."""
        reader = copy(self)
        reader._verification = ArtifactVerification()
        try:
            yield reader
        finally:
            reader._verification.close()
            reader._verification = None

    def inspect(self, *, runtime: bool = False) -> dict[str, Any]:
        """Return schema readiness without initializing or recovering storage.

        Offline inspection preserves source/sidecar bytes using a private copy.
        ``runtime=True`` reads a coordinated committed SQLite transaction instead
        of copying an actively written store. It never initializes or recovers it.
        Artifact verification remains the separate explicit ``verify`` API.
        """
        if not self.meta_path.exists():
            return {
                "status": "uninitialized",
                "schema_version": None,
                "reason": "metadata_missing",
            }
        if not self.meta_path.is_file():
            return {
                "status": "incompatible",
                "schema_version": None,
                "reason": "metadata_not_file",
            }
        try:
            with open_metadata_snapshot(self.meta_path, runtime=runtime) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    )
                }
                if not tables:
                    return {
                        "status": "uninitialized",
                        "schema_version": None,
                        "reason": "metadata_empty",
                    }
                required = {
                    "bt_store_schema": {"version"},
                    "bt_evaluations": {"id", "owner_id", "manifest_json", "created_at"},
                    "bt_chapters": {
                        "evaluation_id",
                        "section",
                        "manifest_json",
                        "invalid_reason",
                        "created_at",
                    },
                    "bt_shared": {
                        "owner_id",
                        "component",
                        "input_identity",
                        "receipt_identity",
                        "manifest_json",
                        "created_at",
                    },
                    "bt_cleanup_receipts": {"id", "manifest_json", "completed_json"},
                }
                if not set(required).issubset(tables) or any(
                    not columns.issubset(
                        {
                            row[1]
                            for row in connection.execute(f"PRAGMA table_info({table})")
                        }
                    )
                    for table, columns in required.items()
                ):
                    return {
                        "status": "incompatible",
                        "schema_version": None,
                        "reason": "schema_incompatible",
                    }
                versions = connection.execute(
                    "SELECT version FROM bt_store_schema"
                ).fetchall()
                version = versions[0][0] if len(versions) == 1 else None
                if version != 1:
                    return {
                        "status": "incompatible",
                        "schema_version": version,
                        "reason": "schema_incompatible",
                    }
                return {"status": "ready", "schema_version": version, "reason": None}
        except (sqlite3.DatabaseError, OSError):
            return {
                "status": "incompatible",
                "schema_version": None,
                "reason": "metadata_unreadable",
            }

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(_path(self.meta_path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def _read_connection(self):
        connection = sqlite3.connect(
            self.meta_path.as_uri() + "?mode=ro", uri=True, timeout=30
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.rollback()
            connection.close()

    def initialize(self) -> None:
        """Create a fresh store; incompatible stores are never migrated."""
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='bt_store_schema'"
            ).fetchone()
            if exists is None:
                other = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if other:
                    raise RuntimeError(
                        "BT metadata is incompatible; use a fresh database"
                    )
                connection.executescript(_SCHEMA)
                connection.execute(
                    "CREATE TABLE bt_table_index(table_id TEXT PRIMARY KEY,"
                    "reference_json TEXT NOT NULL,schema_json TEXT NOT NULL)"
                )
            elif (
                connection.execute("SELECT version FROM bt_store_schema").fetchone()[0]
                != 1
            ):
                raise RuntimeError("BT metadata is incompatible; back up and rebuild")
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")

    def begin(
        self,
        owner_id: str,
        *,
        settings: Mapping[str, Any],
        through: date | str,
        inputs: Mapping[str, Any] | None = None,
        mode: str | None = None,
    ) -> str:
        """Register one numerical evaluation independently of application aliases."""
        _identifier(owner_id, "owner_id")
        end = date.fromisoformat(str(through)).isoformat()
        if mode is not None and mode not in {"signal", "weights", "execution"}:
            raise ValueError("mode must be signal, weights or execution")
        payload = {
            "schema": "bt.evaluation.v1",
            "owner_id": owner_id,
            "settings": _normalize_inputs(settings),
            "through": end,
            "inputs": _normalize_inputs(inputs or {}),
            "mode": mode,
        }
        identity = _identity(payload)
        record = {
            **payload,
            "id": identity,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO bt_evaluations VALUES (?,?,?,?)",
                (identity, owner_id, _json(record), record["created_at"]),
            )
        self.select(identity)
        return identity

    def describe(self, evaluation_id: str) -> dict[str, Any] | None:
        if not self.meta_path.is_file():
            return None
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM bt_evaluations WHERE id=?", (evaluation_id,)
            ).fetchone()
        return (
            None
            if row is None
            else _decode_record(
                row,
                "bt.evaluation.v1",
                id=row["id"],
                owner_id=row["owner_id"],
                created_at=row["created_at"],
            )
        )

    def select(self, evaluation_id: str) -> dict[str, Any]:
        """Resolve an exact retained evaluation without changing a current pointer."""
        record = self.describe(evaluation_id)
        if record is None:
            raise KeyError("unknown BT evaluation")
        return record

    def history(self, owner_id: str) -> list[dict[str, Any]]:
        if not self.meta_path.is_file():
            return []
        with self._read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM bt_evaluations WHERE owner_id=? "
                "ORDER BY created_at DESC,id",
                (owner_id,),
            ).fetchall()
        return [
            _decode_record(
                row,
                "bt.evaluation.v1",
                id=row["id"],
                owner_id=row["owner_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def _table_path(self, table_id: str) -> Path:
        if len(table_id) != 64 or any(c not in "0123456789abcdef" for c in table_id):
            raise ValueError("invalid BT table reference")
        path = self.artifact_path / "tables" / (table_id + ".parquet")
        linked = any(
            parent.is_symlink() or parent.is_junction()
            for parent in (path, path.parent)
        )
        if linked or not path.resolve().is_relative_to(self.artifact_path):
            raise ValueError("BT table reference escapes the artifact root")
        return path

    def _save_table(self, frame: pl.DataFrame, check_canceled: Callable[[], None]):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("BT tables must be Polars DataFrames")
        digest = hash_dataframe(frame)
        registered = self._registered_table(digest)
        if registered is not None:
            return registered
        path = self._table_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        check_canceled()
        existing = path.exists()
        if not path.exists():
            temporary = path.with_name(digest + "." + uuid.uuid4().hex + ".tmp.parquet")
            try:
                frame.write_parquet(_path(temporary), compression="lz4")
                check_canceled()
                try:
                    os.link(_path(temporary), _path(path))
                except FileExistsError:
                    existing = True
                    registered = self._registered_table(digest)
                    if registered is not None:
                        return registered
            finally:
                temporary.unlink(missing_ok=True)
        if existing:
            # Only a registered descriptor authorizes a metadata-only hit.
            # An orphan left by interruption must prove its canonical values.
            try:
                prior = pl.read_parquet(_path(path))
            except pl.exceptions.PolarsError as error:
                raise ValueError("unregistered BT artifact is unreadable") from error
            if hash_dataframe(prior) != digest:
                raise ValueError("unregistered BT artifact differs from publication")
        reference = {
            "backend": "bt",
            "table_id": digest,
            "checksum": _file_hash(path),
            "logical_hash": digest,
            "rows": frame.height,
        }
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS bt_table_index(table_id TEXT PRIMARY KEY,"
                "reference_json TEXT NOT NULL,schema_json TEXT NOT NULL)"
            )
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='bt_table_index'"
            ).fetchone():
                connection.execute(
                    "INSERT OR IGNORE INTO bt_table_index VALUES(?,?,?)",
                    (
                        digest,
                        _json(reference),
                        self._index_schema(reference, frame.schema),
                    ),
                )
        return reference

    @staticmethod
    def _index_schema(reference, schema):
        values = {name: str(dtype) for name, dtype in schema.items()}
        return _json(
            {
                "schema": values,
                "digest": _identity(
                    {
                        "version": "bt.table.index.v2",
                        "reference": reference,
                        "schema": values,
                    }
                ),
            }
        )

    def _registered_table(self, table_id: str, *, original=False):
        """Resolve original owner records, including index-free historical stores."""
        with self._read_connection() as connection:
            if (
                not original
                and connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='bt_table_index'"
                ).fetchone()
            ):
                row = connection.execute(
                    "SELECT reference_json,schema_json FROM bt_table_index "
                    "WHERE table_id=?",
                    (table_id,),
                ).fetchone()
                if row is not None:
                    reference, index = json.loads(row[0]), json.loads(row[1])
                    if (
                        reference.get("table_id") != table_id
                        or reference.get("logical_hash") != table_id
                    ):
                        raise ValueError(
                            "BT table descriptor index source binding differs"
                        )
                    if index.get("digest") != _identity(
                        {
                            "version": "bt.table.index.v2",
                            "reference": reference,
                            "schema": index.get("schema"),
                        }
                    ):
                        raise ValueError("corrupt BT table descriptor index")
                    return reference
            for table in ("bt_shared", "bt_chapters"):
                row = connection.execute(
                    f"SELECT j.value FROM {table},json_each(manifest_json,'$.files') j "
                    "WHERE json_extract(j.value,'$.table_id')=? LIMIT 1",
                    (table_id,),
                ).fetchone()
                if row is not None:
                    return json.loads(row[0])
        return None

    def describe_reference(self, reference: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve a registered immutable table without reading its numerical bytes."""
        if reference.get("backend") != "bt":
            raise ValueError("reference is not a BT table")
        self._table_path(str(reference.get("table_id", "")))
        if reference.get("logical_hash") != reference.get("table_id"):
            raise ValueError("BT table logical identity differs")
        self._table_path(str(reference.get("table_id", "")))
        original = self._registered_table(str(reference.get("table_id", "")))
        if original is None or original != dict(reference):
            raise ValueError("BT table reference differs from its registered record")
        return original

    def index_plan(self) -> dict[str, Any]:
        """Describe explicit table-descriptor maintenance from saved receipts."""
        tables = {}
        if not self.meta_path.is_file():
            return {
                "version": "bt.table.index.v2",
                "meta_path": str(self.meta_path),
                "tables": [],
            }
        with self._read_connection() as connection:
            for table in ("bt_shared", "bt_chapters"):
                for row in connection.execute(f"SELECT manifest_json FROM {table}"):
                    receipt = json.loads(row[0])
                    _verify_record(receipt, receipt["schema"])
                    for ref in receipt["files"].values():
                        tables[ref["table_id"]] = ref
        return {
            "version": "bt.table.index.v2",
            "meta_path": str(self.meta_path),
            "tables": [tables[key] for key in sorted(tables)],
        }

    def build_index(
        self, plan, *, check_canceled=lambda: None, progress=lambda *_: None
    ):
        """Explicitly audit historical tables and register derived descriptors."""
        if plan.get("version") != "bt.table.index.v2" or plan.get("meta_path") != str(
            self.meta_path
        ):
            raise ValueError("BT index plan belongs to another authority/version")
        if not plan["tables"]:
            return {"status": "complete", "tables": 0, "unknown_tables": 0}
        from bagelquant_core.resources import admit_parquet_materialization

        completed, unknown = 0, 0
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS bt_table_index(table_id TEXT PRIMARY KEY,"
                "reference_json TEXT NOT NULL,schema_json TEXT NOT NULL)"
            )
        for position, ref in enumerate(plan["tables"], 1):
            check_canceled()
            if self._registered_table(ref["table_id"], original=True) != ref:
                raise ValueError("BT index plan reference changed")
            path = self._table_path(ref["table_id"])
            if _file_hash(path) != ref["checksum"]:
                raise ValueError("corrupt BT artifact")
            if not admit_parquet_materialization(_path(path)):
                unknown += 1
                progress(position, len(plan["tables"]))
                continue
            frame = pl.read_parquet(_path(path))
            if hash_dataframe(frame) != ref["table_id"]:
                raise ValueError("BT table logical identity differs")
            schema = self._index_schema(ref, frame.schema)
            del frame
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                check_canceled()
                connection.execute(
                    "INSERT OR REPLACE INTO bt_table_index VALUES(?,?,?)",
                    (ref["table_id"], _json(ref), schema),
                )
            progress(position, len(plan["tables"]))
            completed += 1
        return {
            "status": "partial" if unknown else "complete",
            "tables": completed,
            "unknown_tables": unknown,
        }

    def read_reference(self, reference: Mapping[str, Any]) -> pl.DataFrame:
        """Verify bytes and logical identity before decoding an opaque table ref."""
        if self._verification is None:
            with self.read_context() as reader:
                return reader.read_reference(reference)
        reference = self.describe_reference(reference)
        path = self._table_path(str(reference.get("table_id", "")))
        try:
            frame = pl.read_parquet(_path(path))
        except pl.exceptions.PolarsError as error:
            raise ValueError("corrupt or unreadable BT artifact") from error
        if frame.height != reference.get("rows"):
            raise ValueError("BT table row count differs")
        with self._read_connection() as connection:
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='bt_table_index'"
            ).fetchone():
                row = connection.execute(
                    "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                    (reference["table_id"],),
                ).fetchone()
                if row is not None and json.loads(row[0])["schema"] != {
                    name: str(dtype) for name, dtype in frame.schema.items()
                }:
                    raise ValueError("BT table typed schema differs")
        return frame

    def verify_reference(
        self, reference: Mapping[str, Any], *, deep: bool = False
    ) -> None:
        """Check an opaque table without materializing its numerical values."""
        if reference.get("backend") != "bt":
            raise ValueError("reference is not a BT table")
        original = self._registered_table(
            str(reference.get("table_id", "")), original=True
        )
        indexed = self._registered_table(str(reference.get("table_id", "")))
        if indexed != dict(reference) or (
            original is not None and original != dict(reference)
        ):
            raise ValueError("BT descriptor differs from original publication")
        path = self._table_path(str(reference.get("table_id", "")))
        if reference.get("logical_hash") != reference.get("table_id"):
            raise ValueError("BT table logical identity differs")
        if (
            not isinstance(reference.get("rows"), int)
            or isinstance(reference.get("rows"), bool)
            or reference["rows"] < 0
        ):
            raise ValueError("BT table row count is invalid")
        if not path.is_file():
            raise ValueError("corrupt or missing BT artifact")
        verification = self._verification or ArtifactVerification()
        verification.verify(
            path,
            str(reference.get("checksum")),
            receipt=str(reference.get("table_id")),
            checksum=_file_hash,
        )
        if deep:
            frame = self.read_reference(reference)
            if hash_dataframe(frame) != reference.get("logical_hash"):
                raise ValueError("BT table logical identity differs")

    def verify_receipt(self, receipt: Mapping[str, Any]) -> None:
        """Verify all referenced tables, including those not selected for display."""
        _verify_record(receipt, receipt.get("schema"))
        for reference in receipt.get("files", {}).values():
            self.verify_reference(reference)

    def shared_receipt(
        self,
        owner_id: str,
        component: str,
        *,
        input_identity: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.meta_path.is_file():
            return None
        query = "SELECT * FROM bt_shared WHERE owner_id=? AND component=?"
        arguments = [owner_id, component]
        if input_identity is not None:
            query += " AND input_identity=?"
            arguments.append(input_identity)
        query += " ORDER BY created_at DESC,receipt_identity LIMIT 1"
        with self._read_connection() as connection:
            row = connection.execute(query, arguments).fetchone()
        return (
            None
            if row is None
            else _decode_record(
                row,
                "bt.shared.v1",
                owner_id=row["owner_id"],
                component=row["component"],
                input_identity=row["input_identity"],
                receipt_identity=row["receipt_identity"],
            )
        )

    def shared_evaluation(
        self,
        owner_id: str,
        component: str,
        *,
        input_identity: str | None = None,
        names: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, pl.DataFrame]] | None:
        receipt = self.shared_receipt(
            owner_id, component, input_identity=input_identity
        )
        if receipt is None:
            return None
        return receipt, {
            name: self.read_reference(ref)
            for name, ref in receipt["files"].items()
            if names is None or name in names
        }

    def _publication_files(self, frames, file_references, check_canceled):
        files = {}
        for name in dict.fromkeys([*frames, *(file_references or {})]):
            if not name.replace("_", "").isalnum():
                raise ValueError("invalid BT primitive name")
            check_canceled()
            ref = (file_references or {}).get(name)
            if ref is not None:
                ref = self.describe_reference(ref)
                if (
                    name in frames
                    and hash_dataframe(frames[name]) != ref["logical_hash"]
                ):
                    raise ValueError(
                        "BT primitive reference does not match supplied rows"
                    )
                files[name] = dict(ref)
            else:
                files[name] = self._save_table(frames[name], check_canceled)
        return files

    def publish_shared(
        self,
        owner_id: str,
        component: str,
        input_identity: str,
        frames: Mapping[str, pl.DataFrame],
        *,
        metadata: Mapping[str, Any] | None = None,
        sources: Mapping[str, Any] | None = None,
        expected_shared: Mapping[str, Any] | None = None,
        file_references: Mapping[str, Mapping[str, Any]] | None = None,
        check_canceled: Callable[[], None] = lambda: None,
    ) -> dict[str, Any]:
        """Publish an immutable shared primitive set using an explicit CAS proof."""
        for value, name in (
            (owner_id, "owner_id"),
            (component, "component"),
            (input_identity, "input_identity"),
        ):
            _identifier(value, name)
        files = self._publication_files(frames, file_references, check_canceled)
        receipt = {
            "schema": "bt.shared.v1",
            "owner_id": owner_id,
            "component": component,
            "input_identity": input_identity,
            "files": files,
            "metadata": dict(metadata or {}),
            "sources": dict(sources or {}),
        }
        receipt = json.loads(_json(receipt))
        receipt["receipt_identity"] = _identity(receipt)
        check_canceled()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            check_canceled()
            row = connection.execute(
                "SELECT * FROM bt_shared WHERE owner_id=? AND component=? "
                "ORDER BY created_at DESC,receipt_identity LIMIT 1",
                (owner_id, component),
            ).fetchone()
            observed = (
                None
                if row is None
                else _decode_record(
                    row,
                    "bt.shared.v1",
                    owner_id=owner_id,
                    component=component,
                    input_identity=row["input_identity"],
                    receipt_identity=row["receipt_identity"],
                )
            )
            expected = (
                None if expected_shared is None else json.loads(_json(expected_shared))
            )
            if observed != expected:
                if (
                    observed is not None
                    and observed["receipt_identity"] == receipt["receipt_identity"]
                ):
                    return observed
                raise RuntimeError("BT shared primitives changed during publication")
            previous = connection.execute(
                "SELECT * FROM bt_shared WHERE owner_id=? AND component=? "
                "AND input_identity=? ORDER BY created_at DESC LIMIT 1",
                (owner_id, component, input_identity),
            ).fetchone()
            if previous:
                old = _decode_record(
                    previous,
                    "bt.shared.v1",
                    owner_id=owner_id,
                    component=component,
                    input_identity=input_identity,
                    receipt_identity=previous["receipt_identity"],
                )
                if any(files.get(name) != ref for name, ref in old["files"].items()):
                    raise RuntimeError("immutable shared primitive identity disagrees")
                if old["receipt_identity"] == receipt["receipt_identity"]:
                    return old
            _verify_record(receipt, "bt.shared.v1")
            connection.execute(
                "INSERT INTO bt_shared VALUES (?,?,?,?,?,?)",
                (
                    owner_id,
                    component,
                    input_identity,
                    receipt["receipt_identity"],
                    _json(receipt),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return receipt

    def chapter(
        self, evaluation_id: str, section: str, *, verify: bool = False
    ) -> dict[str, Any]:
        """Read published status; optional full verification includes all tables."""
        if not self.meta_path.is_file():
            return {"status": "missing", "section": section}
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM bt_chapters WHERE evaluation_id=? AND section=?",
                (evaluation_id, section),
            ).fetchone()
        if row is None:
            return {"status": "missing", "section": section}
        try:
            receipt = _decode_record(
                row,
                "bt.chapter.v1",
                evaluation_id=evaluation_id,
                section=section,
                updated_at=row["created_at"],
            )
            self.select(evaluation_id)
            if row["invalid_reason"]:
                return {
                    "status": "invalid",
                    "section": section,
                    "reason": row["invalid_reason"],
                }
            if verify:
                self.verify_receipt(receipt)
        except (ValueError, OSError, pl.exceptions.PolarsError):
            return {
                "status": "invalid",
                "section": section,
                "reason": "retained_artifact_corrupt",
            }
        return {**receipt, "status": "ready"}

    def publish_chapter(
        self,
        evaluation_id: str,
        section: str,
        frames: Mapping[str, pl.DataFrame],
        *,
        metadata: Mapping[str, Any] | None = None,
        file_references: Mapping[str, Mapping[str, Any]] | None = None,
        check_canceled: Callable[[], None] = lambda: None,
    ) -> dict[str, Any]:
        """Publish a complete chapter; failures retain prior receipts."""
        self.select(evaluation_id)
        _identifier(section, "section")
        if set(metadata or {}) & {
            "schema",
            "evaluation_id",
            "section",
            "files",
            "receipt_identity",
            "updated_at",
            "status",
        }:
            raise ValueError("chapter metadata uses reserved receipt fields")
        files = self._publication_files(frames, file_references, check_canceled)
        receipt = {
            **dict(metadata or {}),
            "schema": "bt.chapter.v1",
            "evaluation_id": evaluation_id,
            "section": section,
            "files": files,
        }
        receipt = json.loads(_json(receipt))
        receipt["receipt_identity"] = _identity(receipt)
        receipt["updated_at"] = datetime.now(UTC).isoformat()
        check_canceled()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            check_canceled()
            previous = connection.execute(
                "SELECT * FROM bt_chapters WHERE evaluation_id=? AND section=?",
                (evaluation_id, section),
            ).fetchone()
            if previous is not None:
                old = _decode_record(
                    previous,
                    "bt.chapter.v1",
                    evaluation_id=evaluation_id,
                    section=section,
                    updated_at=previous["created_at"],
                )
                if old["receipt_identity"] != receipt["receipt_identity"]:
                    raise RuntimeError("immutable BT chapter already differs")
                return old
            _verify_record(receipt, "bt.chapter.v1")
            connection.execute(
                "INSERT INTO bt_chapters VALUES (?,?,?,NULL,?)",
                (evaluation_id, section, _json(receipt), receipt["updated_at"]),
            )
        return receipt

    def read_chapter(
        self,
        evaluation_id: str,
        section: str,
        *,
        names: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
        if self._verification is None:
            with self.read_context() as reader:
                return reader.read_chapter(evaluation_id, section, names=names)
        receipt = self.chapter(evaluation_id, section, verify=False)
        if receipt["status"] != "ready":
            raise ValueError(
                f"BT chapter unavailable: {receipt.get('reason', receipt['status'])}"
            )
        selected = None if names is None else set(names)
        return receipt, {
            name: self.read_reference(ref)
            for name, ref in receipt["files"].items()
            if selected is None or name in selected
        }

    def invalidate(
        self,
        *,
        reason: str,
        evaluation_ids: Sequence[str] | None = None,
        owner_id: str | None = None,
        sections: Sequence[str] | None = None,
    ) -> int:
        """Mark selected results stale without altering immutable historical bytes."""
        _identifier(reason, "reason")
        clauses, arguments = [], [reason]
        if evaluation_ids is not None:
            if not evaluation_ids:
                return 0
            clauses.append(
                "evaluation_id IN (" + ",".join("?" for _ in evaluation_ids) + ")"
            )
            arguments.extend(evaluation_ids)
        if owner_id is not None:
            clauses.append(
                "evaluation_id IN (SELECT id FROM bt_evaluations WHERE owner_id=?)"
            )
            arguments.append(owner_id)
        if sections is not None:
            if not sections:
                return 0
            clauses.append("section IN (" + ",".join("?" for _ in sections) + ")")
            arguments.extend(sections)
        query = "UPDATE bt_chapters SET invalid_reason=?"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._connect() as connection:
            return connection.execute(query, arguments).rowcount

    def inventory(self) -> dict[str, Any]:
        """Return committed receipts and opaque references without backend paths."""
        if not self.meta_path.is_file():
            return {"evaluations": [], "chapters": [], "shared": [], "tables": []}
        with self._read_connection() as connection:
            evaluations = [
                _decode_record(
                    row,
                    "bt.evaluation.v1",
                    id=row["id"],
                    owner_id=row["owner_id"],
                    created_at=row["created_at"],
                )
                for row in connection.execute(
                    "SELECT * FROM bt_evaluations ORDER BY id"
                )
            ]
            chapters = [
                _decode_record(
                    row,
                    "bt.chapter.v1",
                    evaluation_id=row["evaluation_id"],
                    section=row["section"],
                    updated_at=row["created_at"],
                )
                for row in connection.execute(
                    "SELECT * FROM bt_chapters ORDER BY evaluation_id,section"
                )
            ]
            shared = [
                _decode_record(
                    row,
                    "bt.shared.v1",
                    owner_id=row["owner_id"],
                    component=row["component"],
                    input_identity=row["input_identity"],
                    receipt_identity=row["receipt_identity"],
                )
                for row in connection.execute(
                    "SELECT * FROM bt_shared ORDER BY owner_id,component,input_identity"
                )
            ]
        tables = {
            ref["table_id"]: ref
            for receipt in [*chapters, *shared]
            for ref in receipt["files"].values()
        }
        return {
            "evaluations": evaluations,
            "chapters": chapters,
            "shared": shared,
            "tables": [tables[key] for key in sorted(tables)],
        }

    def verify(self, *, deep: bool = True) -> dict[str, Any]:
        """Inspect every retained receipt/table without repairing or recomputing."""
        issues = []
        tables = {}
        if not self.meta_path.is_file():
            return {"status": "complete", "tables": 0, "issues": []}
        with self._read_connection() as connection:
            for table, schema in (
                ("bt_evaluations", "bt.evaluation.v1"),
                ("bt_chapters", "bt.chapter.v1"),
                ("bt_shared", "bt.shared.v1"),
            ):
                for row in connection.execute(f"SELECT * FROM {table}"):
                    keys = {
                        name: row[name]
                        for name in (
                            "id",
                            "owner_id",
                            "component",
                            "input_identity",
                            "receipt_identity",
                            "evaluation_id",
                            "section",
                        )
                        if name in row.keys()
                    }
                    if table == "bt_evaluations":
                        keys["created_at"] = row["created_at"]
                    elif table == "bt_chapters":
                        keys["updated_at"] = row["created_at"]
                    token = (
                        row["id"]
                        if table == "bt_evaluations"
                        else row["receipt_identity"]
                        if table == "bt_shared"
                        else row["evaluation_id"] + ":" + row["section"]
                    )
                    try:
                        receipt = _decode_record(row, schema, **keys)
                        for reference in receipt.get("files", {}).values():
                            tables[_identity(reference)] = reference
                    except (ValueError, KeyError, TypeError) as error:
                        issues.append(
                            {
                                "table_id": "receipt:" + token,
                                "kind": "receipt",
                                "reason": str(error),
                            }
                        )
        for reference in tables.values():
            try:
                if deep:
                    self.verify_reference(reference, deep=True)
                elif not self._table_path(reference["table_id"]).is_file():
                    raise ValueError("missing BT artifact")
            except (ValueError, OSError, pl.exceptions.PolarsError) as error:
                issues.append({"table_id": reference["table_id"], "reason": str(error)})
        return {
            "status": "corrupt" if issues else "complete",
            "tables": len({reference["table_id"] for reference in tables.values()}),
            "issues": issues,
        }

    def recover(self) -> dict[str, Any]:
        """Report committed integrity and unreferenced candidates; never adopt files."""
        integrity = self.verify()
        metadata_corrupt = any(
            issue.get("kind") == "receipt" for issue in integrity["issues"]
        )
        return {
            **integrity,
            "unreferenced": []
            if metadata_corrupt
            else self.cleanup_plan()["candidates"],
        }

    def cleanup_plan(
        self,
        *,
        table_ids: Sequence[str] | None = None,
        created_before: datetime | None = None,
    ) -> dict[str, Any]:
        """Freeze only unreferenced known table files; retain all committed history."""
        inventory = self.inventory()
        retained = {ref["table_id"] for ref in inventory["tables"]}
        scope = None if table_ids is None else set(table_ids)
        if created_before is not None and created_before.tzinfo is None:
            raise ValueError("cleanup timestamp must be timezone-aware")
        candidates = []
        directory = self.artifact_path / "tables"
        if directory.is_symlink():
            raise ValueError("BT table directory is a symbolic link")
        if directory.exists():
            for path in sorted(directory.iterdir()):
                table_id = path.name.removesuffix(".parquet")
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.suffix != ".parquet"
                    or len(table_id) != 64
                    or any(c not in "0123456789abcdef" for c in table_id)
                    or table_id in retained
                ):
                    continue
                stat = path.stat()
                if scope is not None and table_id not in scope:
                    continue
                if (
                    created_before is not None
                    and stat.st_mtime > created_before.timestamp()
                ):
                    continue
                candidates.append(
                    {
                        "table_id": table_id,
                        "checksum": _file_hash(path),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "ctime_ns": stat.st_ctime_ns,
                    }
                )
        payload = {
            "schema": "bt.cleanup.v1",
            "retained": sorted(retained),
            "candidates": candidates,
            "table_ids": None if scope is None else sorted(scope),
            "created_before": None
            if created_before is None
            else created_before.isoformat(),
        }
        return {**payload, "id": _identity(payload)}

    def cleanup(
        self,
        plan: Mapping[str, Any],
        *,
        check_canceled: Callable[[], None] = lambda: None,
    ) -> dict[str, Any]:
        """Delete exactly a frozen orphan inventory, with an idempotent receipt."""
        payload = {key: value for key, value in plan.items() if key != "id"}
        if _identity(payload) != plan.get("id"):
            raise ValueError("BT cleanup plan identity differs")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT manifest_json,completed_json FROM bt_cleanup_receipts "
                "WHERE id=?",
                (plan["id"],),
            ).fetchone()
            if existing is None:
                cutoff = (
                    datetime.fromisoformat(plan["created_before"])
                    if plan.get("created_before")
                    else None
                )
                if self.cleanup_plan(
                    table_ids=plan.get("table_ids"), created_before=cutoff
                ) != dict(plan):
                    raise RuntimeError(
                        "BT cleanup inventory changed; create a new plan"
                    )
                connection.execute(
                    "INSERT INTO bt_cleanup_receipts VALUES (?,?,?)",
                    (plan["id"], _json(plan), "[]"),
                )
                completed = []
            else:
                if json.loads(existing[0]) != dict(plan):
                    raise ValueError("BT cleanup receipt differs")
                completed = json.loads(existing[1])
        for candidate in plan["candidates"]:
            if candidate["table_id"] in completed:
                continue
            check_canceled()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                retained = {ref["table_id"] for ref in self.inventory()["tables"]}
                if candidate["table_id"] in retained:
                    raise RuntimeError("BT cleanup candidate is now referenced")
                path = self._table_path(candidate["table_id"])
                if path.exists():
                    stat = path.stat()
                    if (
                        stat.st_size != candidate["size"]
                        or stat.st_mtime_ns != candidate["mtime_ns"]
                        or stat.st_ctime_ns != candidate["ctime_ns"]
                        or _file_hash(path) != candidate["checksum"]
                    ):
                        raise RuntimeError("BT cleanup candidate changed")
                    path.unlink()
                completed.append(candidate["table_id"])
                connection.execute(
                    "UPDATE bt_cleanup_receipts SET completed_json=? WHERE id=?",
                    (_json(completed), plan["id"]),
                )
        return {"id": plan["id"], "status": "complete", "deleted": len(completed)}
