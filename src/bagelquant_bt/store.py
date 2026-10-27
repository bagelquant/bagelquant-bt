"""BT-owned immutable evaluation tables, publication and recovery evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import wraps
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import TYPE_CHECKING, Any

import polars as pl
from bagelquant_core import ArtifactVerification
from bagelquant_core.hashing import hash_dataframe
from bagelquant_core.inspection import open_metadata_snapshot

if TYPE_CHECKING:
    from .input_maintenance import SharedInputEvidence

_TABLE_ID_LITERAL = re.compile(r'"([0-9a-f]{64})"')


def _publication_timed(stage):
    def decorate(function):
        @wraps(function)
        def invoke(self, *args, **kwargs):
            started = perf_counter()
            failed = False
            try:
                return function(self, *args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                if self.publication_observer is not None:
                    try:
                        self.publication_observer(stage, perf_counter() - started)
                    except Exception:
                        if not failed:
                            raise

        return invoke

    return decorate


@dataclass
class _PublicationTables:
    references: dict[str, Any]
    frontier: tuple[int, int] | None = None


_SCHEMA = """
CREATE TABLE bt_store_schema(version INTEGER NOT NULL);
INSERT INTO bt_store_schema VALUES(2);
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


def _verify_record(record: Mapping[str, Any], schema: str | None, **keys) -> None:
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


class _FrameReadCache:
    """Operation-local immutable frame memo, bounded by an explicit byte cap."""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.frames = OrderedDict()
        self.lock = RLock()

    def get(self, key, stamp):
        with self.lock:
            item = self.frames.get(key)
            if item is None:
                return None
            if item[1] != stamp:
                self.bytes -= self.frames.pop(key)[2]
                return None
            self.frames.move_to_end(key)
            return item[0].clone()

    def put(self, key, frame, stamp):
        size = frame.estimated_size()
        if size > self.max_bytes or not self.max_bytes:
            return
        with self.lock:
            old = self.frames.pop(key, None)
            if old is not None:
                self.bytes -= old[2]
            while self.frames and self.bytes + size > self.max_bytes:
                _, item = self.frames.popitem(last=False)
                self.bytes -= item[2]
            self.frames[key] = (frame.clone(), stamp, size)
            self.bytes += size

    def close(self):
        with self.lock:
            self.frames.clear()
            self.bytes = 0


def _equal_ordered_prefix(before, current, axis):
    """Find the first unequal complete date with one exact ordered comparison."""
    if (
        not before.height
        or not before[axis].is_sorted()
        or not current[axis].is_sorted()
    ):
        return None
    common = min(before.height, current.height)
    old_rows = before.head(common).select(pl.struct(pl.all())).to_series()
    new_rows = current.head(common).select(pl.struct(pl.all())).to_series()
    unequal = (~old_rows.eq_missing(new_rows)).arg_true()
    if unequal.len():
        position = int(unequal[0])
        first = min(before[axis][position], current[axis][position])
    elif before.height != current.height:
        larger = before if before.height > common else current
        first = larger[axis][common]
    else:
        return before[axis][-1]
    prefix = before[axis].filter(before[axis] < first)
    return prefix[-1] if prefix.len() else None


def _table_evidence(frame, *, axes=None):
    """Compact commutative typed row summaries, captured at publication only."""
    from .table_composition import COORDINATES

    if frame.schema.get("time") != pl.Date or not (
        "factor_return" in frame.columns
        or ("gross_return" in frame.columns and "asset_id" not in frame.columns)
        or "specific_return" in frame.columns
    ):
        return None
    axes = (
        axes
        if axes is not None
        else [name for name in COORDINATES if name in frame.columns]
    )
    if len(set(axes)) != len(axes) or any(
        name not in COORDINATES or name not in frame.columns for name in axes
    ):
        raise ValueError("BT summary evidence axes differ from original table columns")
    canonical = frame.select(sorted(frame.columns))
    rows = (
        frame.select(axes)
        .with_columns(
            canonical.hash_rows(seed=0, seed_1=0, seed_2=0, seed_3=0).alias("_a"),
            canonical.hash_rows(seed=41, seed_1=67, seed_2=97, seed_3=131).alias("_b"),
        )
        .group_by(axes)
        .agg(pl.len().alias("_rows"), pl.col("_a").sum(), pl.col("_b").sum())
        .sort(axes)
    )
    groups = [
        [
            *(str(value) if isinstance(value, date) else value for value in row[:-3]),
            row[-3],
            format(row[-2], "016x"),
            format(row[-1], "016x"),
        ]
        for row in rows.iter_rows()
    ]
    factors = {}
    if {"factor", "family"}.issubset(frame.columns):
        for name, family, first in (
            frame.group_by("factor", "family").agg(pl.col("time").min()).iter_rows()
        ):
            if name in factors and factors[name]["family"] != family:
                return None
            factors[name] = {"family": family, "first_seen": str(first)}
    return {
        "schema": "bt.reference.evidence.v1",
        "axes": axes,
        "groups": groups,
        "factors": factors,
    }


class BTStore:
    """An explicit SQLite authority with immutable, content-addressed Parquet.

    Caller metadata and source receipt references are opaque. No input discovery,
    source verification, governance, numerical execution or machine policy occurs
    here. Table references disclose no path and are read only through this API.
    """

    def storage_usage(self, *, within_root: Path | None = None) -> dict[str, Any]:
        """Inventory metadata, numerical/index files and unknown owned files."""
        from bagelquant_core.storage_usage import store_usage

        return store_usage(self.meta_path, self.artifact_path, within_root=within_root)

    def __init__(
        self,
        meta_path: str | Path,
        artifact_path: str | Path,
        *,
        publication_observer: Callable[[str, float], None] | None = None,
    ):
        self.meta_path = Path(meta_path).resolve()
        self.artifact_path = Path(artifact_path).resolve()
        self.publication_observer = publication_observer
        self._verification: ArtifactVerification | None = None
        self._frame_cache: _FrameReadCache | None = None
        from .index_blocks import MetadataBlocks

        self._metadata = MetadataBlocks(self.artifact_path)

    @contextmanager
    def read_context(self, *, cache_bytes: int = 64 * 1024 * 1024):
        """Share proofs and a bounded frame memo for one finite operation.

        The caller may lower/disable the explicit byte cap. Cached frames expire
        on exit/failure and file-stat changes; they never establish input proof
        or survive as another numerical authority.
        """
        if (
            isinstance(cache_bytes, bool)
            or not isinstance(cache_bytes, int)
            or cache_bytes < 0
        ):
            raise ValueError("BT frame cache bytes must be a nonnegative integer")
        reader = copy(self)
        reader._verification = ArtifactVerification()
        reader._frame_cache = _FrameReadCache(cache_bytes - cache_bytes // 4)
        from .index_blocks import MetadataBlocks

        reader._metadata = MetadataBlocks(
            self.artifact_path, cache_bytes=cache_bytes // 4
        )
        try:
            yield reader
        finally:
            reader._verification.close()
            reader._verification = None
            reader._frame_cache.close()
            reader._frame_cache = None
            reader._metadata.close()

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
                if version != 2:
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
    def _connect(self, *, initializing=False):
        connection = sqlite3.connect(
            self.meta_path.as_uri() + ("?mode=rwc" if initializing else "?mode=rw"),
            uri=True,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                if not initializing:
                    self._require_schema(connection)
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
            self._require_schema(connection)
            yield connection
        finally:
            connection.rollback()
            connection.close()

    @staticmethod
    def _require_schema(connection):
        try:
            versions = connection.execute(
                "SELECT version FROM bt_store_schema"
            ).fetchall()
        except sqlite3.DatabaseError as error:
            raise ValueError(
                "BT metadata is incompatible; initialize a fresh store"
            ) from error
        if len(versions) != 1 or versions[0][0] != 2:
            raise ValueError("BT metadata schema 2 required; initialize a fresh store")

    def initialize(self) -> None:
        """Create a fresh store; incompatible stores are never migrated."""
        if self.meta_path.is_file() and self.meta_path.stat().st_size:
            with self._read_connection():
                pass
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        with self._connect(initializing=True) as connection:
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
                != 2
            ):
                raise RuntimeError("BT metadata is incompatible; back up and rebuild")
        with self._connect(initializing=True) as connection:
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

    def _save_table(
        self,
        frame: pl.DataFrame,
        check_canceled: Callable[[], None],
        *,
        _prepared=None,
    ):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("BT tables must be Polars DataFrames")
        if _prepared is None:
            digest = hash_dataframe(frame)
            registered = self._registered_table(digest)
        else:
            digest, lookup = _prepared
            registered = self._fresh_publication_reference(
                digest, lookup, check_canceled
            )
        if registered is not None:
            if not self._table_path(digest).is_file():
                raise ValueError(
                    "Registered BT artifact is missing; explicit recovery is required"
                )
            return registered
        path = self._table_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        check_canceled()
        existing = path.exists()
        if existing and _prepared is not None:
            # A writer may have published since the batch's read transaction.
            # Resolve its original before treating an existing file as an orphan.
            registered = self._registered_table(digest)
            if registered is not None:
                if not path.is_file():
                    raise ValueError(
                        "Registered BT artifact is missing; "
                        "explicit recovery is required"
                    )
                return registered
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
            connection.execute("BEGIN IMMEDIATE")
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
                        self._index_schema(reference, frame.schema, frame=frame),
                    ),
                )
        return reference

    @_publication_timed("indexing")
    def _index_schema(self, reference, schema, *, frame=None, coordinates=None):
        from .index_blocks import coordinate_frame, evidence_frame

        values = {name: str(dtype) for name, dtype in schema.items()}
        payload: dict[str, Any] = {"schema": values, "columns": list(schema)}
        if frame is not None:
            facts = coordinate_frame(frame)
            payload["coordinates"] = (
                None if facts is None else self._metadata.save(facts)
            )
            evidence = evidence_frame(frame)
            if evidence is not None:
                header, groups = evidence
                payload["evidence"] = {**header, "groups": self._metadata.save(groups)}
        elif coordinates is not None:
            if coordinates == {"composition": True}:
                payload["coordinates"] = coordinates
            else:
                from .table_composition import COORDINATES, _coordinate_frame

                facts = (
                    _coordinate_frame(coordinates, values)
                    if coordinates
                    else pl.DataFrame(
                        schema={
                            name: dtype
                            for name, dtype in schema.items()
                            if name in COORDINATES
                        }
                        | {"_rows": pl.UInt32}
                    )
                )
                # Legacy explicit maintenance callers pass serialized dates.
                facts = facts.with_columns(
                    pl.col(name).str.to_date()
                    for name, dtype in values.items()
                    if dtype == "Date"
                    and name in facts.columns
                    and facts.schema[name] == pl.String
                )
                payload["coordinates"] = self._metadata.save(facts)
        return self._seal_index(reference, payload)

    @staticmethod
    def _seal_index(reference, payload):
        return _json(
            {
                **payload,
                "digest": _identity(
                    {"version": "bt.table.index.v3", "reference": reference, **payload}
                ),
            }
        )

    def _selected_count(self, reference, index, predicates=(), keys=None):
        coordinate_index = index.get("coordinates")
        if coordinate_index is None:
            raise ValueError(
                "BT selected references need explicit coordinate-index maintenance"
            )
        if reference.get("format") == "concat":
            count = 0
            for part in reference["parts"]:
                with self._registered_table_read(
                    part["reference"]["table_id"]
                ) as child:
                    if child["index"] is None:
                        raise ValueError(
                            "BT composition needs explicit table-index maintenance"
                        )
                    count += self._selected_count(
                        child["reference"],
                        child["index"],
                        [*part["filters"], *predicates],
                        keys,
                    )
            return count
        return self._metadata.count(coordinate_index, predicates=predicates, keys=keys)

    def _registered_table(self, table_id: str, *, original=False):
        """Resolve the same canonical original record without retaining its index."""
        with self._registered_table_read(table_id, original=original) as registered:
            return registered["reference"]

    @contextmanager
    def _registered_table_read(self, table_id: str, *, original=False, _originals=None):
        """Validate one row in T1; close its transaction before yielding.

        Raw serialization and the decoded index belong only to this call. The
        public descriptor and numerical reader use this one validation path.
        """
        registered = {"reference": None, "index": None, "raw": None}
        row = reference = index = None
        try:
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
                        if "columns" in index and (
                            not isinstance(index["columns"], list)
                            or any(
                                not isinstance(name, str) for name in index["columns"]
                            )
                            or len(set(index["columns"])) != len(index["columns"])
                            or set(index["columns"]) != set(index.get("schema", {}))
                        ):
                            raise ValueError("corrupt BT original-column index")
                        if (
                            reference.get("table_id") != table_id
                            or reference.get("logical_hash") != table_id
                        ):
                            raise ValueError(
                                "BT table descriptor index source binding differs"
                            )
                        if index.get("digest") != _identity(
                            {
                                "version": "bt.table.index.v3",
                                "reference": reference,
                                "schema": index.get("schema"),
                                **(
                                    {"columns": index["columns"]}
                                    if "columns" in index
                                    else {}
                                ),
                                **(
                                    {"coordinates": index["coordinates"]}
                                    if "coordinates" in index
                                    else {}
                                ),
                                **(
                                    {"evidence": index["evidence"]}
                                    if "evidence" in index
                                    else {}
                                ),
                            }
                        ):
                            raise ValueError("corrupt BT table descriptor index")
                        registered.update(
                            reference=reference, index=index, raw=(row[0], row[1])
                        )
                if registered["reference"] is None:
                    originals = (
                        self._original_table_references(connection, {table_id})
                        if _originals is None
                        else _originals
                    )
                    registered["reference"] = originals.get(table_id)
            # Do not retain sqlite rows or duplicate parsed aliases across yield.
            row = reference = index = None
            yield registered
        finally:
            row = reference = index = None
            registered.clear()

    @staticmethod
    def _original_table_references(connection, table_ids, check_canceled=lambda: None):
        """Find requested originals in one streamed pass, without decoding others.

        Literal tokens only reject impossible candidates. Any Unicode escape
        keeps a receipt eligible; only an exact files-subtree match authenticated
        by its original receipt supplies authority. Nothing survives this call.
        """
        remaining = set(table_ids)
        references = {}
        row = raw = receipt = matched = value = None
        pending = []
        returned = False
        try:
            if not remaining:
                returned = True
                return references
            singleton = next(iter(remaining)) if len(remaining) == 1 else None
            for table in ("bt_shared", "bt_chapters"):
                check_canceled()
                query, parameters = f"SELECT manifest_json FROM {table}", ()
                if singleton is not None:
                    # Scalar reads filter in SQLite instead of copying unrelated
                    # receipt bodies through Python for one requested ID.
                    query += (
                        " WHERE instr(manifest_json,?)>0 OR instr(manifest_json,?)>0"
                    )
                    parameters = (singleton, "\\u")
                for row in connection.execute(query, parameters):
                    check_canceled()
                    raw = row[0]
                    if "\\u" not in raw and not any(
                        match[1] in remaining
                        for match in _TABLE_ID_LITERAL.finditer(raw)
                    ):
                        continue
                    receipt = json.loads(raw)
                    matched = {}
                    pending = [iter((receipt.get("files", {}),))]
                    while pending:
                        check_canceled()
                        try:
                            value = next(pending[-1])
                        except StopIteration:
                            pending.pop()
                            continue
                        if isinstance(value, dict):
                            identifier = value.get("table_id")
                            if isinstance(identifier, str) and identifier in remaining:
                                matched.setdefault(identifier, value)
                            pending.append(iter(value.values()))
                        elif isinstance(value, list):
                            pending.append(iter(value))
                    if matched:
                        _verify_record(receipt, receipt["schema"])
                        references.update(matched)
                        remaining.difference_update(matched)
                        if not remaining:
                            returned = True
                            return references
            returned = True
            return references
        finally:
            # Release only owned aliases, including iterators, on every exit.
            # Foreign exception frames and returned descriptors remain intact.
            row = raw = receipt = matched = value = None
            pending.clear()
            remaining.clear()
            if not returned:
                references.clear()

    @staticmethod
    def _receipt_frontier(connection):
        # Owner APIs append immutable receipts. Invalidation changes only
        # invalid_reason; cleanup never removes historical receipt rows.
        return tuple(
            connection.execute(
                f"SELECT coalesce(max(rowid),0) FROM {table}"
            ).fetchone()[0]
            for table in ("bt_shared", "bt_chapters")
        )

    @_publication_timed("lookup")
    def _publication_table_references(self, table_ids, check_canceled):
        """Resolve indexed hits normally, then all misses in one finite read."""
        references = dict.fromkeys(table_ids)
        lookup = _PublicationTables(references)
        for identifier in references:
            check_canceled()
            # Reuse the descriptor validator while deferring original lookup.
            with self._registered_table_read(identifier, _originals=references) as row:
                references[identifier] = row["reference"]
        missing = {key for key, value in references.items() if value is None}
        if missing:
            with self._read_connection() as connection:
                references.update(
                    self._original_table_references(connection, missing, check_canceled)
                )
                lookup.frontier = self._receipt_frontier(connection)
        return lookup

    def _fresh_publication_reference(self, identifier, lookup, check_canceled):
        references = lookup.references
        if references[identifier] is not None:
            return references[identifier]
        # A concurrent publisher may have registered this original, even if its
        # numerical file is now absent. Never recreate it from a stale miss.
        with self._registered_table_read(identifier, _originals=references) as row:
            references[identifier] = row["reference"]
        if references[identifier] is None:
            with self._read_connection() as connection:
                frontier = self._receipt_frontier(connection)
                if frontier != lookup.frontier:
                    references.update(
                        self._original_table_references(
                            connection,
                            {key for key, ref in references.items() if ref is None},
                            check_canceled,
                        )
                    )
                    lookup.frontier = frontier
        return references[identifier]

    def _fresh_reference_index(self, registered):
        """Observe T2/T3 afresh; reuse decoding only for the exact T1 row.

        Index maintenance may replace rows. Any appearance, disappearance or
        serialization change during this read fails closed, including cache hits.
        No transaction/snapshot or index survives the call.
        """
        row = None
        try:
            with self._read_connection() as connection:
                row = (
                    connection.execute(
                        "SELECT reference_json,schema_json FROM bt_table_index "
                        "WHERE table_id=?",
                        (registered["reference"]["table_id"],),
                    ).fetchone()
                    if connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE name='bt_table_index'"
                    ).fetchone()
                    else None
                )
                if (row is None) != (registered["raw"] is None) or (
                    row is not None
                    and (
                        row[0] != registered["raw"][0] or row[1] != registered["raw"][1]
                    )
                ):
                    raise ValueError("BT table index changed during read")
            return registered["index"]
        finally:
            row = None

    def describe_reference(self, reference: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve a registered immutable table without reading its numerical bytes."""
        with self._describe_reference_read(reference) as registered:
            return registered["reference"]

    @contextmanager
    def _describe_reference_read(self, reference, *, _originals=None):
        if reference.get("backend") != "bt":
            raise ValueError("reference is not a BT table")
        self._table_path(str(reference.get("table_id", "")))
        if reference.get("logical_hash") != reference.get("table_id"):
            raise ValueError("BT table logical identity differs")
        if reference.get("format") == "concat":
            if (
                reference["table_id"]
                != _identity({"format": "concat", "parts": reference["parts"]})
                or reference["checksum"] != reference["table_id"]
                or any(
                    part["reference"].get("format") is not None
                    for part in reference["parts"]
                )
                or reference["rows"] != sum(part["rows"] for part in reference["parts"])
            ):
                raise ValueError("BT composition metadata identity differs")
        self._table_path(str(reference.get("table_id", "")))
        with self._registered_table_read(
            str(reference.get("table_id", "")), _originals=_originals
        ) as registered:
            if registered["reference"] is None or registered["reference"] != dict(
                reference
            ):
                raise ValueError(
                    "BT table reference differs from its registered record"
                )
            yield registered

    def publish_table(
        self, frame: pl.DataFrame, *, check_canceled=lambda: None
    ) -> dict[str, Any]:
        """Publish one new primitive through the same immutable table authority."""
        check_canceled()
        return self._save_table(frame, check_canceled)

    def publish_continuation(
        self,
        frames: Mapping[str, pl.DataFrame],
        *,
        previous: Mapping[str, Mapping[str, Any]],
        output_start: date | None = None,
        output_end: date | None = None,
        check_canceled: Callable[[], None] = lambda: None,
    ) -> dict[str, dict[str, Any]]:
        """Retain equal historical prefixes; write only changed primitive tails.

        Supplying an admitted ``output_start`` declares delta frames; optional
        ``output_end`` retains the unaffected suffix after a finite repair. This
        owner composes original indexed references without reading old values.
        The caller's numerical plan/source proofs establish the replaced range.

        Otherwise original tables are consumed through the typed reader. No old
        artifact is rehashed or rewritten. Summary tables without a daily axis
        are published through the ordinary immutable publisher.
        """
        if self._verification is None:
            with self.read_context() as reader:
                return reader.publish_continuation(
                    frames,
                    previous=previous,
                    output_start=output_start,
                    output_end=output_end,
                    check_canceled=check_canceled,
                )
        if output_start is not None:
            return self._publish_delta(
                frames,
                previous=previous,
                output_start=output_start,
                output_end=output_end,
                check_canceled=check_canceled,
            )
        if output_end is not None:
            raise ValueError("BT continuation output end requires its output start")
        references = {}
        for name, frame in frames.items():
            check_canceled()
            original = previous.get(name)
            if original is None:
                references[name] = self.publish_table(
                    frame, check_canceled=check_canceled
                )
                continue
            old = self.read_reference(original)
            if old.schema == frame.schema and old.equals(frame):
                references[name] = self.describe_reference(original)
                continue
            axis = next(
                (
                    key
                    for key in ("evaluation_date", "time", "target_date")
                    if frame.schema.get(key) == pl.Date
                ),
                None,
            )
            if (
                axis is None
                or frame.schema != old.schema
                or frame[axis].null_count()
                or old[axis].null_count()
            ):
                references[name] = self.publish_table(
                    frame, check_canceled=check_canceled
                )
                continue
            grouping = (
                [
                    key
                    for key in ("window_kind", "window_id", "method")
                    if key in frame.columns
                ]
                if "window_id" in frame.columns
                else ["path_kind", "lag"]
                if "path_kind" in frame.columns and "lag" in frame.columns
                else ["lag"]
                if "lag" in frame.columns
                else ["quantile"]
                if "quantile" in frame.columns
                else []
            )
            while True:
                groups = (
                    frame.partition_by(grouping, maintain_order=True)
                    if grouping and frame.height
                    else [frame]
                )
                if pl.concat(groups).equals(frame):
                    break
                grouping = grouping[:-1]
            parts = []
            for group in groups:
                predicates = (
                    [
                        {"column": key, "operation": "eq", "value": group[key][0]}
                        for key in grouping
                    ]
                    if group.height
                    else []
                )
                from .table_composition import select

                before = select(old, predicates)
                boundary = _equal_ordered_prefix(before, group, axis)
                if boundary is not None:
                    parts.append(
                        {
                            "reference": original,
                            "filters": [
                                *predicates,
                                {"column": axis, "operation": "le", "value": boundary},
                            ],
                        }
                    )
                delta = (
                    group if boundary is None else group.filter(pl.col(axis) > boundary)
                )
                if delta.height or not parts:
                    parts.append(
                        {
                            "reference": self.publish_table(
                                delta, check_canceled=check_canceled
                            )
                        }
                    )
            references[name] = self.compose_reference(parts)
        return references

    def _publish_delta(
        self, frames, *, previous, output_start, output_end, check_canceled
    ):
        if type(output_start) is not date or (
            output_end is not None
            and (type(output_end) is not date or output_end < output_start)
        ):
            raise ValueError("BT continuation needs ordered Date output bounds")
        # Validate the complete admitted batch before saving any delta. Resolve
        # new IDs together: a scalar publication per primitive scans retained
        # receipts repeatedly when its optional descriptor is absent.
        axes = {}
        for name, frame in frames.items():
            check_canceled()
            axis = next(
                (
                    key
                    for key in ("evaluation_date", "time", "target_date")
                    if frame.schema.get(key) == pl.Date
                ),
                None,
            )
            if (
                axis is None
                or frame[axis].null_count()
                or (
                    frame.height
                    and (
                        frame[axis].min() < output_start
                        or (output_end is not None and frame[axis].max() > output_end)
                    )
                )
            ):
                raise ValueError(
                    "BT delta rows must lie inside their admitted output interval"
                )
            axes[name] = axis
        # Continuation keys label returned references; they have never been
        # shared/chapter primitive names. Preserve that public mapping contract.
        names = {
            name: f"delta_{index}"
            for index, (name, frame) in enumerate(frames.items())
            if frame.height or name not in previous
        }
        published = self._publication_files(
            {internal: frames[name] for name, internal in names.items()},
            None,
            check_canceled,
        )
        references = {}
        for name, frame in frames.items():
            check_canceled()
            axis = axes[name]
            parts = []
            original = previous.get(name)
            if original is not None:
                parts.append(
                    {
                        "reference": original,
                        "filters": [
                            {
                                "column": axis,
                                "operation": "le",
                                "value": output_start - timedelta(days=1),
                            }
                        ],
                    }
                )
            if frame.height or original is None:
                parts.append({"reference": published[names[name]]})
            if original is not None and output_end is not None:
                parts.append(
                    {
                        "reference": original,
                        "filters": [
                            {
                                "column": axis,
                                "operation": "ge",
                                "value": output_end + timedelta(days=1),
                            }
                        ],
                    }
                )
            references[name] = self.compose_reference(parts)
        return references

    @_publication_timed("composition")
    def compose_reference(self, parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Compose original references and exact counts, never expanded grids."""
        from .table_composition import COORDINATES, filters

        selected, schema, complete_facts = [], None, True
        for part in parts:
            reference = self.describe_reference(part["reference"])
            with self._registered_table_read(reference["table_id"]) as registered:
                index = registered["index"]
                if index is None or "columns" not in index:
                    raise ValueError(
                        "BT composition needs explicit original-column index "
                        "maintenance"
                    )
                if schema is None:
                    schema = {name: index["schema"][name] for name in index["columns"]}
                elif schema != index["schema"]:
                    raise ValueError("BT composition table schemas differ")
                predicates = filters(
                    part.get("filters", ()),
                    {
                        key: value
                        for key, value in index["schema"].items()
                        if key in COORDINATES
                    },
                )
            members = reference.get("parts", [{"reference": reference, "filters": []}])
            for member in members:
                with self._registered_table_read(
                    member["reference"]["table_id"]
                ) as registered:
                    original, index = registered["reference"], registered["index"]
                    if index is None:
                        raise ValueError(
                            "BT composition needs explicit coordinate-index maintenance"
                        )
                    combined = [*member["filters"], *predicates]
                    rows = (
                        original["rows"]
                        if not combined
                        else self._selected_count(original, index, combined)
                    )
                    complete_facts = (
                        complete_facts and index.get("coordinates") is not None
                    )
                    if rows or not selected:
                        selected.append(
                            {"reference": original, "filters": combined, "rows": rows}
                        )
        if schema is None:
            raise ValueError("BT composition needs at least one registered table")
        payload = {"format": "concat", "parts": selected}
        identity = _identity(payload)
        reference = {
            "backend": "bt",
            "table_id": identity,
            "logical_hash": identity,
            "checksum": identity,
            "rows": sum(item["rows"] for item in selected),
            **payload,
        }
        header = self._index_schema(
            reference,
            schema,
            coordinates={"composition": True} if complete_facts else None,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_publication_files(
                {str(i): part["reference"] for i, part in enumerate(selected)}
            )
            connection.execute(
                "INSERT OR IGNORE INTO bt_table_index VALUES(?,?,?)",
                (identity, _json(reference), header),
            )
        return self.describe_reference(reference)

    def reference_evidence(
        self,
        reference: Mapping[str, Any],
        *,
        time_column: str = "time",
        filters: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any] | None:
        """Read compact date content and causal factor catalog from owner records.

        Unknown original summaries return None. This never decodes a numerical
        table or implicitly prepares legacy indexes.
        """
        _identifier(time_column, "time column")
        from .table_composition import COORDINATES
        from .table_composition import filters as validate_filters

        with self._describe_reference_read(reference) as registered:
            index = self._fresh_reference_index(registered)
            if index is None:
                return None
            predicates = validate_filters(
                filters,
                {
                    key: value
                    for key, value in index["schema"].items()
                    if key in COORDINATES
                },
            )
        return self._reference_evidence(reference, predicates, time_column=time_column)

    def reference_interval_identity(
        self,
        reference: Mapping[str, Any],
        *,
        start: date,
        end: date,
        cutoff: date | None = None,
        time_column: str = "time",
    ) -> str | None:
        """Fragmentation-independent typed contents in an explicit date window.

        Factor availability is clipped when that column exists. A gross path
        without it must be bounded by the caller's complete economic endpoint.
        """
        if (
            type(start) is not date
            or type(end) is not date
            or start > end
            or (cutoff is not None and type(cutoff) is not date)
        ):
            raise ValueError("BT interval identity needs ordered Date bounds")
        _identifier(time_column, "time column")
        evidence = self._reference_evidence(
            reference,
            (
                {"column": time_column, "operation": "ge", "value": str(start)},
                {"column": time_column, "operation": "le", "value": str(end)},
            ),
            cutoff=cutoff,
            time_column=time_column,
        )
        return (
            None
            if evidence is None
            else _identity(
                {
                    "schema": evidence["schema"],
                    "time_column": time_column,
                    "start": str(start),
                    "end": str(end),
                    "dates": evidence["dates"],
                }
            )
        )

    def _reference_evidence(
        self, reference, predicates, *, cutoff=None, time_column="time"
    ):
        from .evidence_groups import evidence_groups

        with self._metadata.fence(), evidence_groups() as groups:
            return self._collect_reference_evidence(
                reference, predicates, groups, cutoff=cutoff, time_column=time_column
            )

    def _collect_reference_evidence(
        self, reference, predicates, groups, *, cutoff=None, time_column="time"
    ):
        from .evidence_groups import add_groups, date_identities

        reference = self.describe_reference(reference)
        parts = reference.get("parts", ({"reference": reference, "filters": []},))
        catalog, schema = {}, None
        axes = None
        for part in parts:
            with self._describe_reference_read(part["reference"]) as registered:
                index = self._fresh_reference_index(registered)
                evidence = None if index is None else index.get("evidence")
                if (
                    evidence is None
                    or index["schema"].get(time_column) != "Date"
                    or time_column not in evidence["axes"]
                ):
                    return None
                if schema is not None and (
                    schema != index["schema"] or axes != evidence["axes"]
                ):
                    raise ValueError("BT evidence composition schema differs")
                schema, axes = index["schema"], evidence["axes"]
                selected = [*part["filters"], *predicates]
                if cutoff is not None and "available_date" in axes:
                    selected.append(
                        {
                            "column": "available_date",
                            "operation": "le",
                            "value": str(cutoff),
                        }
                    )
                if any(item["column"] not in axes for item in selected):
                    return None

                def records(evidence=evidence, selected=selected, axes=axes):
                    for group in self._metadata.groups(evidence, selected):
                        values = dict(zip(axes, group[:-3], strict=True))
                        keep = True
                        for item in selected:
                            value, limit = values[item["column"]], item["value"]
                            if hasattr(limit, "isoformat"):
                                limit = limit.isoformat()
                            if value is None:
                                keep = (
                                    item["column"] == "available_date"
                                    and cutoff is not None
                                )
                            else:
                                keep = (
                                    value == limit
                                    if item["operation"] == "eq"
                                    else value >= limit
                                    if item["operation"] == "ge"
                                    else value <= limit
                                )
                            if not keep:
                                break
                        if keep:
                            key = group[:-3]
                            yield (
                                _json(key),
                                key[axes.index(time_column)],
                                *group[-3:],
                            )

                add_groups(groups, records())
                for name, item in evidence["factors"].items():
                    if name in catalog and catalog[name]["family"] != item["family"]:
                        raise ValueError("BT factor catalogue family differs")
                    if (
                        name not in catalog
                        or item["first_seen"] < catalog[name]["first_seen"]
                    ):
                        catalog[name] = dict(item)
                self._fresh_reference_index(registered)
        if axes is None:
            return None
        return {
            "schema": schema,
            "time_column": time_column,
            "factors": catalog,
            "dates": date_identities(groups, axes),
            "date_rows": dict(
                groups.execute(
                    "SELECT day,sum(n) FROM groups GROUP BY day ORDER BY day"
                )
            ),
        }

    def index_plan(
        self,
        *,
        evidence: bool = False,
        references: Sequence[Mapping[str, Any]] | None = None,
        time_column: str = "time",
    ) -> dict[str, Any]:
        """Describe explicit table-descriptor maintenance from saved receipts."""
        tables = {}
        if not self.meta_path.is_file():
            return {
                "version": "bt.table.index.v3",
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
                        for part in ref.get("parts", ()):
                            original = part["reference"]
                            tables[original["table_id"]] = original
        if references is not None:
            requested = set()
            for ref in references:
                requested.add(ref["table_id"])
                requested.update(
                    part["reference"]["table_id"] for part in ref.get("parts", ())
                )
            tables = {key: ref for key, ref in tables.items() if key in requested}
        if evidence:
            with self._read_connection() as connection:
                needed = {}
                has_index = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='bt_table_index'"
                ).fetchone()
                for key, ref in tables.items():
                    if ref.get("format") == "concat":
                        continue
                    row = (
                        connection.execute(
                            "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                            (key,),
                        ).fetchone()
                        if has_index
                        else None
                    )
                    index = None if row is None else json.loads(row[0])
                    if (
                        index is None
                        or index.get("evidence") is None
                        or time_column not in index["evidence"]["axes"]
                    ):
                        needed[key] = ref
                tables = needed
        return {
            "version": "bt.table.index.v3",
            "meta_path": str(self.meta_path),
            "tables": [tables[key] for key in sorted(tables)],
        }

    def maintain_reference(
        self, reference: Mapping[str, Any], *, check_canceled=lambda: None
    ) -> dict[str, Any]:
        """Explicitly audit/index one retained registered table and its leaves.

        This maintenance entry does not publish a replacement receipt or change
        original bytes. Ordinary selected reads never call it implicitly.
        """
        reference = self.describe_reference(reference)
        originals = [part["reference"] for part in reference.get("parts", ())]
        # Original leaves precede concatenations so bounded typed selection has
        # native coordinate authority when the composed index is rebuilt.
        tables = {item["table_id"]: item for item in originals}
        tables[reference["table_id"]] = reference
        return self.build_index(
            {
                "version": "bt.table.index.v3",
                "meta_path": str(self.meta_path),
                "tables": list(tables.values()),
            },
            check_canceled=check_canceled,
        )

    def build_index(
        self, plan, *, check_canceled=lambda: None, progress=lambda *_: None
    ):
        """Explicitly audit historical tables and register derived descriptors."""
        if plan.get("version") != "bt.table.index.v3" or plan.get("meta_path") != str(
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
            original = self._registered_table(ref["table_id"], original=True)
            if (original is not None and original != ref) or self._registered_table(
                ref["table_id"]
            ) != ref:
                raise ValueError("BT index plan reference changed")
            paths = (
                [
                    self._table_path(part["reference"]["table_id"])
                    for part in ref["parts"]
                ]
                if ref.get("format") == "concat"
                else [self._table_path(ref["table_id"])]
            )
            if not all(admit_parquet_materialization(_path(path)) for path in paths):
                unknown += 1
                progress(position, len(plan["tables"]))
                continue
            if ref.get("format") == "concat":
                self.verify_reference(ref, deep=True)
                # Leaves are the sole coordinate authority. Explicit maintenance
                # must not republish a composed historical grid.
                with self._registered_table_read(ref["table_id"]) as registered:
                    index = registered["index"]
                if index is None:
                    with self._registered_table_read(
                        ref["parts"][0]["reference"]["table_id"]
                    ) as leaf:
                        index = {
                            key: leaf["index"][key] for key in ("schema", "columns")
                        }
                schema = self._seal_index(
                    ref,
                    {
                        **{
                            key: value
                            for key, value in index.items()
                            if key != "digest"
                        },
                        "coordinates": {"composition": True},
                    },
                )
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    check_canceled()
                    self._require_publication_files({"composed": ref})
                    connection.execute(
                        "INSERT OR REPLACE INTO bt_table_index VALUES(?,?,?)",
                        (ref["table_id"], _json(ref), schema),
                    )
                progress(position, len(plan["tables"]))
                completed += 1
                continue
            else:
                path = paths[0]
                if _file_hash(path) != ref["checksum"]:
                    raise ValueError("corrupt BT artifact")
                frame = pl.read_parquet(_path(path))
                if hash_dataframe(frame) != ref["table_id"]:
                    raise ValueError("BT table logical identity differs")
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                check_canceled()
                schema = self._index_schema(ref, frame.schema, frame=frame)
                del frame
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

    def read_reference(
        self,
        reference: Mapping[str, Any],
        *,
        filters: Sequence[Mapping[str, Any]] = (),
        coordinates: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Read registered typed rows with indexed pruning and Parquet pushdown.

        Date/String/Integer coordinate predicates use eq/ge/le. Disjoint
        original leaves are skipped before opening a numerical file. Missing
        coordinate authority requires explicit index maintenance; this reader
        never implicitly decodes history to establish a selection.
        """
        if self._verification is None:
            with self.read_context() as reader:
                return reader.read_reference(
                    reference, filters=filters, coordinates=coordinates
                )
        with (
            self._metadata.fence(),
            self._describe_reference_read(reference) as registered,
        ):
            return self._read_registered_reference(
                registered, filters=filters, coordinates=coordinates
            )

    def _read_registered_reference(self, registered, *, filters, coordinates):
        reference = registered["reference"]
        index = None
        try:
            from .table_composition import (
                COORDINATES,
                admit_coordinates,
                select,
                select_keys,
            )
            from .table_composition import filters as admit_filters

            index = self._fresh_reference_index(registered)
            if (filters or coordinates is not None) and index is None:
                raise ValueError(
                    "BT selected reads need explicit coordinate-index maintenance"
                )
            predicates = (
                []
                if not filters
                else admit_filters(
                    filters,
                    {
                        key: dtype
                        for key, dtype in index["schema"].items()
                        if key in COORDINATES
                    },
                )
            )
            expected = reference["rows"]
            if coordinates is not None:
                coordinates = admit_coordinates(coordinates, index["schema"])
            if predicates or coordinates is not None:
                expected = self._selected_count(
                    reference, index, predicates, coordinates
                )
            if (predicates or coordinates is not None) and expected == 0:
                return self._empty_indexed_table(
                    index["schema"], columns=index.get("columns")
                )
            key = (
                reference["table_id"],
                _json(predicates),
                None if coordinates is None else hash_dataframe(coordinates),
            )
            leaf = reference.get("format") != "concat"
            stamp = self._reference_stamp(reference) if leaf else None
            cached = (
                self._frame_cache.get(key, stamp)
                if leaf and self._frame_cache is not None
                else None
            )
            if cached is not None:
                return cached
            if (
                leaf
                and self._frame_cache is not None
                and (predicates or coordinates is not None)
            ):
                full = self._frame_cache.get(
                    (reference["table_id"], _json([]), None), stamp
                )
                if full is not None:
                    frame = select(full, predicates)
                    if coordinates is not None:
                        frame = select_keys(frame, coordinates)
                    if frame.height != expected:
                        raise ValueError("BT selected memo row count differs")
                    self._frame_cache.put(key, frame, stamp)
                    return frame
            if reference.get("format") == "concat":
                frames = []
                for part in reference["parts"]:
                    combined = [*part["filters"], *predicates]
                    selected = self.read_reference(
                        part["reference"], filters=combined, coordinates=coordinates
                    )
                    if selected.height:
                        frames.append(selected)
                if frames:
                    columns = frames[0].columns
                    frame = pl.concat([item.select(columns) for item in frames])
                    if frame.height != expected:
                        raise ValueError("BT composed table selected row count differs")
                    return frame
                if index is None:
                    raise ValueError(
                        "BT empty selection needs explicit table-index maintenance"
                    )
                return self._empty_indexed_table(
                    index["schema"], columns=index.get("columns")
                )
            path = self._table_path(str(reference.get("table_id", "")))
            try:
                if predicates or coordinates is not None:
                    selected = select(pl.scan_parquet(_path(path)), predicates)
                    if coordinates is not None:
                        selected = select_keys(selected, coordinates)
                    frame = selected.collect()
                else:
                    frame = pl.read_parquet(_path(path))
            except pl.exceptions.PolarsError as error:
                raise ValueError("corrupt or unreadable BT artifact") from error
            if frame.height != expected:
                raise ValueError("BT table row count differs")
            index = self._fresh_reference_index(registered)
            if index is not None and index["schema"] != {
                name: str(dtype) for name, dtype in frame.schema.items()
            }:
                raise ValueError("BT table typed schema differs")
            if self._frame_cache is not None:
                if self._reference_stamp(reference) != stamp:
                    raise ValueError("BT artifact changed during read")
                self._frame_cache.put(key, frame, stamp)
            return frame
        finally:
            # Clear only aliases owned by this reader; preserve exception identity.
            index = None

    def _reference_stamp(self, reference):
        leaves = (
            (part["reference"] for part in reference["parts"])
            if reference.get("format") == "concat"
            else (reference,)
        )
        result = []
        try:
            for leaf in leaves:
                stat = self._table_path(leaf["table_id"]).stat()
                result.append(
                    (
                        leaf["table_id"],
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                        stat.st_ino,
                    )
                )
        except OSError as error:
            raise ValueError("missing or unreadable BT artifact") from error
        return tuple(result)

    @staticmethod
    def _empty_indexed_table(schema, *, columns=None):
        if columns is None:
            raise ValueError(
                "BT empty selection needs explicit original-column index maintenance"
            )
        try:
            return pl.DataFrame(
                schema={key: getattr(pl, schema[key]) for key in (columns or schema)}
            )
        except (AttributeError, TypeError) as error:
            raise ValueError(
                "BT selection needs a supported registered typed schema"
            ) from error

    def execution_references(
        self,
        frames: Mapping[str, pl.DataFrame],
        *,
        previous: Mapping[str, Mapping[str, Any]],
        through: date,
        check_canceled: Callable[[], None] = lambda: None,
    ) -> dict[str, dict[str, Any]]:
        """Append native execution tails without decoding or rewriting history.

        Current lot/P&L snapshots and checkpoint frames replace their previous
        references. Every daily/event table must be strictly after the restored
        boundary. The engine's cumulative checkpoint owns financial summaries.
        """
        snapshots = {"open_lots", "ideal_open_lots", "transaction_pnl"}
        result = dict(previous)
        for name, frame in frames.items():
            check_canceled()
            original = previous.get(name)
            if original is None or name in snapshots or "checkpoint_" in name:
                result[name] = self.publish_table(frame, check_canceled=check_canceled)
                continue
            axis = (
                "exit_date"
                if name == "fifo_matches"
                else ("execution_date" if name == "transaction_plans" else "time")
            )
            if axis not in frame.columns or frame.schema[axis] != pl.Date:
                raise ValueError(
                    f"execution continuation table {name} lacks its dated axis"
                )
            if frame[axis].null_count() or frame.filter(pl.col(axis) <= through).height:
                raise ValueError(
                    f"execution continuation table {name} overlaps completed history"
                )
            if frame.is_empty():
                result[name] = self.compose_reference(
                    [
                        {
                            "reference": original,
                            "filters": [
                                {"column": axis, "operation": "le", "value": through}
                            ],
                        }
                    ]
                )
                continue
            tail = self.publish_table(frame, check_canceled=check_canceled)
            try:
                result[name] = self.compose_reference(
                    [
                        {
                            "reference": original,
                            "filters": [
                                {"column": axis, "operation": "le", "value": through}
                            ],
                        },
                        {"reference": tail},
                    ]
                )
            except ValueError as error:
                raise ValueError(
                    f"execution continuation table {name}: {error}"
                ) from error
        return result

    def verify_reference(
        self, reference: Mapping[str, Any], *, deep: bool = False
    ) -> None:
        """Check an opaque table without materializing its numerical values."""
        if reference.get("backend") != "bt":
            raise ValueError("reference is not a BT table")
        if reference.get("format") == "concat":
            reference = self.describe_reference(reference)
            if (
                _identity({"format": "concat", "parts": reference["parts"]})
                != reference["table_id"]
                or sum(part["rows"] for part in reference["parts"]) != reference["rows"]
            ):
                raise ValueError("BT composition identity differs")
            for part in reference["parts"]:
                self.verify_reference(part["reference"], deep=deep)
            if deep:
                self._verify_coordinates(reference, self.read_reference(reference))
            return
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
            self._verify_coordinates(reference, frame)

    def _verify_coordinates(self, reference, frame):
        from .index_blocks import coordinate_frame, evidence_frame

        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                (reference["table_id"],),
            ).fetchone()
        if row is None:
            return
        index = json.loads(row[0])
        if index.get("columns") != frame.columns:
            raise ValueError(
                "BT original-column index differs from original table columns"
            )
        recorded = index.get("coordinates")
        if recorded is not None and not recorded.get("composition"):
            columns = [name for name in recorded["columns"] if name != "_rows"]
            actual = coordinate_frame(frame.select(columns))
            self._verify_metadata_frame(actual, recorded)
        if index.get("evidence") is not None:
            evidence, groups = evidence_frame(frame, axes=index["evidence"]["axes"])
            if evidence != {
                k: v for k, v in index["evidence"].items() if k != "groups"
            }:
                raise ValueError(
                    "BT numerical summary evidence differs from original table rows"
                )
            self._verify_metadata_frame(groups, index["evidence"]["groups"])

    def _verify_metadata_frame(self, actual, descriptor):
        offset = 0
        for frame in self._metadata.frames(descriptor):
            expected = actual.slice(offset, frame.height)
            # Counts are exact integers; old metadata may have UInt64 counts.
            expected = expected.cast(frame.schema)
            if not frame.equals(expected):
                raise ValueError(
                    "BT coordinate/evidence index differs from original table rows"
                )
            offset += frame.height
        if offset != actual.height:
            raise ValueError("BT coordinate/evidence index row count differs")

    def verify_receipt(self, receipt: Mapping[str, Any]) -> None:
        """Verify all referenced tables, including those not selected for display."""
        _verify_record(receipt, receipt.get("schema"))
        if receipt.get("schema") == "bt.shared.v1":
            from .input_maintenance import verify_binding

            with self._read_connection() as connection:
                verify_binding(connection, receipt)
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
            receipt = (
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
            if receipt is not None:
                from .input_maintenance import verify_binding

                verify_binding(connection, receipt)
        return receipt

    def maintain_shared_inputs(
        self,
        owner_id: str,
        *,
        original: Mapping[str, Any],
        original_inputs: Mapping[str, Any],
        replacement_inputs: Mapping[str, Any],
        settings: Mapping[str, Any],
        kernel: str,
        scope: Mapping[str, Any],
        consumed: Mapping[str, Mapping[str, SharedInputEvidence]],
        original_request: Mapping[str, str],
        report_reference: str,
        report_sha256: str,
        sources: Mapping[str, Any],
        check_canceled: Callable[[], None] = lambda: None,
    ) -> dict[str, Any]:
        """Bind proven unchanged legacy signal primitives at the same boundary.

        Complete source-owned typed projections and the original execution
        report are required. A caller may use a published source-readiness
        report for ordinary same-boundary reuse once original authority has
        been explicitly restored. No numerical values are decoded or computed;
        missing coverage may be explicitly initialized separately. This grants
        no append, group upgrade or causal continuation state.
        """
        from .input_maintenance import maintain

        return maintain(
            self,
            owner_id,
            original=original,
            original_inputs=original_inputs,
            replacement_inputs=replacement_inputs,
            settings=settings,
            kernel=kernel,
            scope=scope,
            consumed=consumed,
            original_request=original_request,
            report_reference=report_reference,
            report_sha256=report_sha256,
            sources=sources,
            check_canceled=check_canceled,
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

    def _publication_files(
        self,
        frames: Mapping[str, pl.DataFrame],
        file_references: Mapping[str, Mapping[str, Any]] | None,
        check_canceled,
    ):
        files = {}
        names: list[str] = [*frames, *(file_references or {})]
        for name in dict.fromkeys(names):
            if not name.replace("_", "").isalnum():
                raise ValueError("invalid BT primitive name")
        hashes = self._publication_hashes(frames, check_canceled)
        lookup = self._publication_table_references(
            [
                *hashes.values(),
                *(
                    str(ref.get("table_id", ""))
                    for ref in (file_references or {}).values()
                ),
            ],
            check_canceled,
        )
        for name in dict.fromkeys(names):
            check_canceled()
            ref = (file_references or {}).get(name)
            if ref is not None:
                with self._describe_reference_read(
                    ref, _originals=lookup.references
                ) as row:
                    ref = row["reference"]
                if name in frames and ref.get("format") == "concat":
                    raise ValueError(
                        "Composed BT publication accepts references only; "
                        "omit the supplied frame"
                    )
                if name in frames and hashes[name] != ref["logical_hash"]:
                    raise ValueError(
                        "BT primitive reference does not match supplied rows"
                    )
                files[name] = dict(ref)
            else:
                digest = hashes[name]
                files[name] = self._save_table(
                    frames[name],
                    check_canceled,
                    _prepared=(digest, lookup),
                )
                lookup.references[digest] = files[name]
        return files

    @_publication_timed("hashing")
    def _publication_hashes(self, frames, check_canceled):
        hashes = {}
        aliases = {}
        for name, frame in frames.items():
            check_canceled()
            if not isinstance(frame, pl.DataFrame):
                raise TypeError("BT tables must be Polars DataFrames")
            key = id(frame)
            if key not in aliases:
                aliases[key] = hash_dataframe(frame)
            hashes[name] = aliases[key]
        return hashes

    def _require_publication_files(self, files):
        """Fence artifact existence against cleanup under the metadata writer."""
        for reference in files.values():
            originals = (
                [part["reference"] for part in reference["parts"]]
                if reference.get("format") == "concat"
                else [reference]
            )
            if any(
                not self._table_path(original["table_id"]).is_file()
                for original in originals
            ):
                raise ValueError(
                    "BT publication artifact is missing; explicit recovery is required"
                )
            with self._read_connection() as connection:
                for original in [reference, *originals]:
                    row = connection.execute(
                        "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                        (original["table_id"],),
                    ).fetchone()
                    if row is not None:
                        for identifier in self._metadata.identifiers(
                            json.loads(row[0])
                        ):
                            if not self._metadata.path(identifier).is_file():
                                raise ValueError(
                                    "BT publication metadata block is missing; "
                                    "explicit recovery required"
                                )

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
        _maintenance_admission: Mapping[str, Any] | None = None,
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
            self._require_publication_files(files)
            from .input_maintenance import verify_binding

            binding = receipt["metadata"].get("primitive_only_input_binding")
            if binding is not None and _maintenance_admission != binding:
                admitted = connection.execute(
                    "SELECT * FROM bt_shared WHERE owner_id=? AND component=? "
                    "AND json_extract(manifest_json,"
                    "'$.metadata.primitive_only_input_binding.identity')=? LIMIT 1",
                    (owner_id, component, binding.get("identity")),
                ).fetchone()
                authority = (
                    None
                    if admitted is None
                    else _decode_record(
                        admitted,
                        "bt.shared.v1",
                        owner_id=owner_id,
                        component=component,
                        input_identity=admitted["input_identity"],
                        receipt_identity=admitted["receipt_identity"],
                    )
                )
                if (
                    authority is None
                    or authority["metadata"].get("primitive_only_input_binding")
                    != binding
                ):
                    raise ValueError(
                        "Primitive binding requires maintain_shared_inputs admission"
                    )
                verify_binding(connection, authority)
            verify_binding(connection, receipt)
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
            self._require_publication_files(files)
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
        for reference in list(tables.values()):
            for part in reference.get("parts", ()):
                original = part["reference"]
                tables[original["table_id"]] = original
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
                else:
                    selected = (
                        [part["reference"] for part in reference.get("parts", ())]
                        if reference.get("format") == "concat"
                        else [reference]
                    )
                    if any(
                        not self._table_path(item["table_id"]).is_file()
                        for item in selected
                    ):
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
        metadata_blocks = self._metadata_cleanup_candidates(
            excluded={row["table_id"] for row in candidates},
            created_before=created_before,
        )
        payload = {
            "schema": "bt.cleanup.v1",
            "retained": sorted(retained),
            "candidates": candidates,
            "metadata_blocks": metadata_blocks,
            "table_ids": None if scope is None else sorted(scope),
            "created_before": None
            if created_before is None
            else created_before.isoformat(),
        }
        return {**payload, "id": _identity(payload)}

    def _metadata_cleanup_candidates(self, *, excluded=(), created_before=None):
        """Keep every block used by a retained descriptor, even shared grids."""
        retained = set()
        if self.meta_path.is_file():
            with self._read_connection() as connection:
                if not connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='bt_table_index'"
                ).fetchone():
                    return []  # Missing authority never authorizes block deletion.
                for row in connection.execute(
                    "SELECT table_id,schema_json FROM bt_table_index"
                ):
                    if row[0] not in excluded:
                        retained.update(self._metadata.identifiers(json.loads(row[1])))
        if not self._metadata.root.exists():
            return []
        if self._metadata.root.is_symlink():
            raise ValueError("BT metadata block directory is a symbolic link")
        candidates = []
        for path in sorted(self._metadata.root.iterdir()):
            identifier = path.stem
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix != ".parquet"
                or len(identifier) != 64
                or any(char not in "0123456789abcdef" for char in identifier)
                or identifier in retained
            ):
                continue
            facts = path.stat()
            if (
                created_before is not None
                and facts.st_mtime > created_before.timestamp()
            ):
                continue
            if _file_hash(path) != identifier:
                continue  # Unknown/corrupt evidence is never a cleanup candidate.
            candidates.append(
                {
                    "block_id": identifier,
                    "size": facts.st_size,
                    "mtime_ns": facts.st_mtime_ns,
                    "ctime_ns": facts.st_ctime_ns,
                }
            )
        return candidates

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
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='bt_table_index'"
                ).fetchone():
                    connection.execute(
                        "DELETE FROM bt_table_index WHERE table_id=?",
                        (candidate["table_id"],),
                    )
                completed.append(candidate["table_id"])
                connection.execute(
                    "UPDATE bt_cleanup_receipts SET completed_json=? WHERE id=?",
                    (_json(completed), plan["id"]),
                )
        for candidate in plan.get("metadata_blocks", ()):
            key = "block:" + candidate["block_id"]
            if key in completed:
                continue
            check_canceled()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT 1 FROM bt_table_index, json_tree(schema_json) j "
                    "WHERE j.key='id' AND j.value=? LIMIT 1",
                    (candidate["block_id"],),
                ).fetchone()
                if row is not None:
                    raise RuntimeError("BT metadata block is now referenced")
                path = self._metadata.path(candidate["block_id"])
                if path.exists():
                    facts = path.stat()
                    if (
                        facts.st_size != candidate["size"]
                        or facts.st_mtime_ns != candidate["mtime_ns"]
                        or facts.st_ctime_ns != candidate["ctime_ns"]
                        or _file_hash(path) != candidate["block_id"]
                    ):
                        raise RuntimeError("BT metadata cleanup candidate changed")
                    path.unlink()
                completed.append(key)
                connection.execute(
                    "UPDATE bt_cleanup_receipts SET completed_json=? WHERE id=?",
                    (_json(completed), plan["id"]),
                )
        return {
            "id": plan["id"],
            "status": "complete",
            "deleted": sum(not item.startswith("block:") for item in completed),
            "metadata_deleted": sum(item.startswith("block:") for item in completed),
        }
