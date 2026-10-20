"""Bounded immutable metadata blocks; numerical Parquet remains the body authority."""

from __future__ import annotations

import hashlib
import io
import os
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from threading import RLock

import polars as pl
from polars import read_parquet as read_metadata_parquet

from . import table_composition
from .table_composition import select, select_keys

MAX_ROWS = 65_536
MAX_BYTES = 8 * 1024 * 1024


def coordinate_frame(frame: pl.DataFrame) -> pl.DataFrame | None:
    columns = [name for name in table_composition.COORDINATES if name in frame.columns]
    if not columns:
        return None
    return (
        frame.group_by(columns)
        .len(name="_rows")
        .with_columns(pl.col("_rows").cast(pl.UInt64))
        .sort(columns)
    )


def evidence_frame(frame: pl.DataFrame, *, axes=None):
    """Capture the same UInt64 row summaries without Python per-row objects."""
    if axes is None and (
        frame.schema.get("time") != pl.Date
        or not (
            "factor_return" in frame.columns
            or ("gross_return" in frame.columns and "asset_id" not in frame.columns)
            or "specific_return" in frame.columns
        )
    ):
        return None
    axes = (
        axes
        if axes is not None
        else [name for name in table_composition.COORDINATES if name in frame.columns]
    )
    canonical = frame.select(sorted(frame.columns))
    groups = (
        frame.select(axes)
        .with_columns(
            canonical.hash_rows(seed=0, seed_1=0, seed_2=0, seed_3=0).alias("_a"),
            canonical.hash_rows(seed=41, seed_1=67, seed_2=97, seed_3=131).alias("_b"),
        )
        .group_by(axes)
        .agg(pl.len().alias("_rows"), pl.col("_a").sum(), pl.col("_b").sum())
        .with_columns(pl.col("_rows").cast(pl.UInt64))
        .sort(axes)
    )
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
        "factors": factors,
    }, groups.select([*axes, "_rows", "_a", "_b"])


def _value(value):
    return value.isoformat() if isinstance(value, date) else value


def _stamp(path):
    facts = path.stat()
    return (
        facts.st_dev,
        facts.st_ino,
        facts.st_size,
        facts.st_mtime_ns,
        facts.st_ctime_ns,
    )


def _bounds(frame):
    return {
        name: {
            "min": _value(frame[name].min()),
            "max": _value(frame[name].max()),
            "nulls": frame[name].null_count(),
        }
        for name, dtype in frame.schema.items()
        if name in table_composition.COORDINATES
        and (dtype in (pl.Date, pl.String) or dtype.is_integer())
    }


class MetadataBlocks:
    """Content-addressed metadata with finite, stat-fenced decoded caching."""

    def __init__(self, artifact_path: Path, *, cache_bytes=0):
        self.root = artifact_path / "indexes"
        self.cache_bytes = cache_bytes
        self.bytes = 0
        self.cache = OrderedDict()
        self.lock = RLock()
        self._fences = ContextVar("bt_metadata_fences", default=())

    @contextmanager
    def fence(self):
        """Recheck every consumed block at the complete operation boundary."""
        consumed = {}
        token = self._fences.set((*self._fences.get(), consumed))
        try:
            yield
            for path, before in consumed.items():
                if path.is_symlink() or _stamp(path) != before:
                    raise ValueError("BT metadata block changed during read")
        except OSError as error:
            raise ValueError("missing or unreadable BT metadata block") from error
        finally:
            self._fences.reset(token)

    def close(self):
        with self.lock:
            self.cache.clear()
            self.bytes = 0

    def path(self, identifier):
        if len(identifier) != 64 or any(
            c not in "0123456789abcdef" for c in identifier
        ):
            raise ValueError("invalid BT metadata block identity")
        path = self.root / (identifier + ".parquet")
        if path.is_symlink() or self.root.is_symlink():
            raise ValueError("BT metadata block path is a symbolic link")
        if path.resolve().parent != self.root.resolve():
            raise ValueError("BT metadata block escapes its owner")
        return path

    def save(self, frame: pl.DataFrame) -> dict:
        """Publish deterministic bounded chunks, deduplicated by encoded bytes."""
        return self.save_batches(frame.iter_slices(MAX_ROWS), schema=frame.schema)

    def save_batches(self, batches: Iterable[pl.DataFrame], *, schema) -> dict:
        descriptor = {
            "columns": list(schema),
            "schema": {k: str(v) for k, v in schema.items()},
            "rows": 0,
            "blocks": [],
        }
        for batch in batches:
            if batch.schema != schema:
                raise ValueError("BT metadata batch schema differs")
            pending = list(batch.iter_slices(MAX_ROWS))
            while pending:
                part = pending.pop(0)
                if part.estimated_size() > MAX_BYTES:
                    if part.height <= 1:
                        raise ValueError("BT metadata row exceeds the block budget")
                    half = part.height // 2
                    pending[0:0] = [part.head(half), part.slice(half)]
                    continue
                if part.is_empty():
                    continue
                output = io.BytesIO()
                part.write_parquet(output, compression="zstd", statistics=True)
                content = output.getvalue()
                digest = hashlib.sha256(content).hexdigest()
                path = self.path(digest)
                self.root.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        raise ValueError("conflicting BT metadata block")
                else:
                    temporary = self.root / (digest + "." + uuid.uuid4().hex + ".tmp")
                    try:
                        with temporary.open("xb") as stream:
                            stream.write(content)
                            stream.flush()
                            os.fsync(stream.fileno())
                        try:
                            os.link(temporary, path)
                        except FileExistsError:
                            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                                raise ValueError(
                                    "conflicting BT metadata block"
                                ) from None
                    finally:
                        temporary.unlink(missing_ok=True)
                descriptor["rows"] += part.height
                descriptor["blocks"].append(
                    {
                        "id": digest,
                        "rows": part.height,
                        "bytes": len(content),
                        "decoded_bytes": part.estimated_size(),
                        "bounds": _bounds(part),
                    }
                )
        if self.root.exists():
            # The following SQLite commit must not outlive directory entries.
            for directory in (self.root, self.root.parent):
                descriptor_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor_fd)
                finally:
                    os.close(descriptor_fd)
        return descriptor

    @staticmethod
    def _disjoint(block, predicates):
        for item in predicates:
            bound = block["bounds"].get(item["column"])
            # Null availability has special evidence semantics. Being
            # conservative here keeps pruning valid for both index consumers.
            if bound is None or bound["min"] is None or bound["nulls"]:
                continue
            value = _value(item["value"])
            op = item["operation"]
            if (
                (op == "eq"
                and (value < bound["min"] or value > bound["max"]))
                or (op == "ge"
                and value > bound["max"])
                or (op == "le"
                and value < bound["min"])
            ):
                return True
        return False

    def _read(self, block, descriptor):
        path = self.path(block["id"])
        try:
            before = _stamp(path)
        except FileNotFoundError as error:
            raise ValueError(
                "missing BT metadata block; explicit recovery required"
            ) from error
        if (
            before[2] != block["bytes"]
            or block["decoded_bytes"] > MAX_BYTES
            or block["rows"] > MAX_ROWS
        ):
            raise ValueError("BT metadata block size differs")
        with self.lock:
            cached = self.cache.get(block["id"])
            if cached is not None and cached[0] == before:
                self.cache.move_to_end(block["id"])
                frame = cached[1]
            else:
                if cached is not None:
                    self.bytes -= self.cache.pop(block["id"])[1].estimated_size()
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != block["id"]:
                    raise ValueError("corrupt BT metadata block")
                frame = read_metadata_parquet(io.BytesIO(content))
                if frame.estimated_size() > MAX_BYTES:
                    raise ValueError("BT metadata block exceeds the decoded budget")
                if _stamp(path) != before:
                    raise ValueError("BT metadata block changed during read")
                size = frame.estimated_size()
                if self.cache_bytes and size <= self.cache_bytes:
                    while self.cache and self.bytes + size > self.cache_bytes:
                        _, old = self.cache.popitem(last=False)
                        self.bytes -= old[1].estimated_size()
                    self.cache[block["id"]] = (before, frame)
                    self.bytes += size
            if (
                frame.height != block["rows"]
                or frame.columns != descriptor["columns"]
                or {k: str(v) for k, v in frame.schema.items()} != descriptor["schema"]
            ):
                raise ValueError("BT metadata block typed schema/count differs")
            if _bounds(frame) != block["bounds"]:
                raise ValueError("BT metadata block pruning bounds differ")
            for consumed in self._fences.get():
                prior = consumed.setdefault(path, before)
                if prior != before:
                    raise ValueError("BT metadata block changed during read")
            return frame.clone()

    def frames(self, descriptor, *, predicates=(), keys=None, apply_filters=True):
        if not isinstance(descriptor, Mapping) or "blocks" not in descriptor:
            raise ValueError("BT selections need explicit coordinate-index maintenance")
        if sum(block["rows"] for block in descriptor["blocks"]) != descriptor["rows"]:
            raise ValueError("BT metadata block counts differ")
        if keys is not None and keys.is_empty():
            return
        key_predicates = (
            []
            if keys is None
            else [
                {"column": name, "operation": op, "value": _value(value)}
                for name in keys.columns
                for op, value in (("ge", keys[name].min()), ("le", keys[name].max()))
            ]
        )
        for block in descriptor["blocks"]:
            if self._disjoint(block, [*predicates, *key_predicates]):
                continue
            frame = self._read(block, descriptor)
            if apply_filters:
                frame = select(frame, predicates)
                if keys is not None:
                    frame = select_keys(frame, keys)
            yield frame

    def count(self, descriptor, *, predicates=(), keys=None):
        return sum(
            int(frame["_rows"].sum() or 0)
            for frame in self.frames(descriptor, predicates=predicates, keys=keys)
        )

    def groups(self, evidence, predicates=()):
        for frame in self.frames(
            evidence["groups"], predicates=predicates, apply_filters=False
        ):
            for row in frame.iter_rows():
                yield [
                    *(_value(value) for value in row[:-3]),
                    row[-3],
                    format(row[-2], "016x"),
                    format(row[-1], "016x"),
                ]

    def identifiers(self, index):
        result = set()
        for descriptor in (
            index.get("coordinates"),
            (index.get("evidence") or {}).get("groups"),
        ):
            if isinstance(descriptor, Mapping):
                result.update(block["id"] for block in descriptor.get("blocks", ()))
        return result
