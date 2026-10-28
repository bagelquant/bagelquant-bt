"""Explicit receipt retirement and exact, shared-reference-safe file deletion."""

from __future__ import annotations

import json
import re
from pathlib import Path

TABLE_KEYS = {
    "bt_evaluations": ("id",),
    "bt_chapters": ("evaluation_id", "section"),
    "bt_shared": ("owner_id", "component", "input_identity", "receipt_identity"),
}


def _rows(connection):
    from .store import _decode_record

    result = {}
    for table, keys in TABLE_KEYS.items():
        rows = [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]
        schema = {
            "bt_evaluations": "bt.evaluation.v1",
            "bt_chapters": "bt.chapter.v1",
            "bt_shared": "bt.shared.v1",
        }[table]
        for row in rows:
            coordinates = {key: row[key] for key in keys}
            coordinates["updated_at" if table == "bt_chapters" else "created_at"] = row[
                "created_at"
            ]
            if table == "bt_shared":
                coordinates.pop("created_at")
            _decode_record(row, schema, **coordinates)
        result[table] = rows
    return result


def _references(rows):
    found = {}

    def visit(value):
        if isinstance(value, dict):
            if value.get("backend") == "bt" and "table_id" in value:
                found[value["table_id"]] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for row in rows:
        visit(json.loads(row["manifest_json"]).get("files", {}))
    return found


def _safe(store, relative):
    if not re.fullmatch(r"(?:tables|indexes)/[a-f0-9]{64}\.parquet", relative):
        raise ValueError("release cannot remove unknown or non-BT files")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("unsafe BT release path")
    path = store.artifact_path / relative
    for member in (path, *path.parents):
        if member == store.artifact_path.parent:
            break
        if member.is_symlink():
            raise ValueError("BT release cannot follow symbolic links")
    if not path.resolve().is_relative_to(store.artifact_path):
        raise ValueError("BT release path escapes its owner")
    return path


def plan_release(store, evaluation_ids, *, owner_ids=()):
    """Preview exact evaluations and unshared owner continuations to retire."""
    from .store import _identity

    evaluations, owners = sorted(set(evaluation_ids)), sorted(set(owner_ids))
    if not store.meta_path.is_file():
        rows = {table: [] for table in TABLE_KEYS}
        indexes = {}
    else:
        with store._read_connection() as db:
            rows = _rows(db)
            indexes = (
                {
                    row[0]: json.loads(row[1])
                    for row in db.execute(
                        "SELECT table_id,schema_json FROM bt_table_index"
                    )
                }
                if db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='bt_table_index'"
                ).fetchone()
                else {}
            )
    selected, kept = {}, {}
    for table, records in rows.items():

        def wanted(row, table=table):
            return (
                row["owner_id"] in owners
                if table == "bt_shared"
                else row["id" if table == "bt_evaluations" else "evaluation_id"]
                in evaluations
            )

        selected[table] = [row for row in records if wanted(row)]
        kept[table] = [row for row in records if not wanted(row)]
    removed = _references([row for records in selected.values() for row in records])
    retained = _references([row for records in kept.values() for row in records])
    table_ids = sorted(set(removed) - set(retained))
    candidates = []
    for identifier in table_ids:
        path = store._table_path(identifier)
        if path.exists():
            relative = path.relative_to(store.artifact_path).as_posix()
            _safe(store, relative)
            candidates.append(
                {
                    "path": relative,
                    "sha256": removed[identifier]["checksum"],
                    "bytes": path.stat().st_size,
                }
            )
    removed_blocks, kept_blocks = set(), set()
    for identifier, index in indexes.items():
        (removed_blocks if identifier in table_ids else kept_blocks).update(
            store._metadata.identifiers(index)
        )
    for identifier in sorted(removed_blocks - kept_blocks):
        path = store._metadata.root / f"{identifier}.parquet"
        if path.is_file():
            relative = path.relative_to(store.artifact_path).as_posix()
            _safe(store, relative)
            candidates.append(
                {"path": relative, "sha256": identifier, "bytes": path.stat().st_size}
            )
    body = {
        "schema": "bt.release.v1",
        "evaluation_ids": evaluations,
        "owner_ids": owners,
        "registry": _identity(rows),
        "records": selected,
        "table_ids": table_ids,
        "files": candidates,
    }
    return {
        **body,
        "id": _identity(body),
        "bytes": sum(item["bytes"] for item in candidates),
    }


def apply_release(store, plan):
    """Archive original receipts before unlinking, with an idempotent retry journal."""
    from .store import _file_hash, _identity, _json

    body = {key: value for key, value in plan.items() if key not in {"id", "bytes"}}
    if body.get("schema") != "bt.release.v1" or _identity(body) != plan.get("id"):
        raise ValueError("BT release plan identity differs")
    store.initialize()
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "CREATE TABLE IF NOT EXISTS bt_release_plans(id TEXT PRIMARY KEY,"
            "plan_json TEXT NOT NULL,completed INTEGER NOT NULL DEFAULT 0,"
            "removed_json TEXT NOT NULL DEFAULT '{}')"
        )
        previous = db.execute(
            "SELECT plan_json,completed,removed_json FROM bt_release_plans WHERE id=?",
            (plan["id"],),
        ).fetchone()
        if previous:
            if json.loads(previous[0]) != dict(plan):
                raise ValueError("BT release request differs")
            if previous[1]:
                return {
                    "id": plan["id"],
                    "bytes": sum(json.loads(previous[2]).values()),
                }
        else:
            if plan_release(
                store, plan["evaluation_ids"], owner_ids=plan["owner_ids"]
            ) != dict(plan):
                raise RuntimeError("BT release inventory changed; review a new preview")
            db.execute(
                "INSERT INTO bt_release_plans VALUES(?,?,0,'{}')",
                (plan["id"], _json(plan)),
            )
            for table, records in plan["records"].items():
                if table not in TABLE_KEYS:
                    raise ValueError("unknown BT release owner")
                for record in records:
                    keys = TABLE_KEYS[table]
                    db.execute(
                        f"DELETE FROM {table} WHERE "
                        + " AND ".join(f"{key}=?" for key in keys),
                        tuple(record[key] for key in keys),
                    )
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='bt_table_index'"
            ).fetchone():
                db.executemany(
                    "DELETE FROM bt_table_index WHERE table_id=?",
                    [(identifier,) for identifier in plan["table_ids"]],
                )
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        has_index = db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='bt_table_index'"
        ).fetchone()
        active_rows = _rows(db)
        retained_tables = _references(
            [row for rows in active_rows.values() for row in rows]
        )
        indexed_tables, retained_blocks = set(), set()
        if has_index:
            for identifier, schema in db.execute(
                "SELECT table_id,schema_json FROM bt_table_index"
            ):
                indexed_tables.add(identifier)
                retained_blocks.update(store._metadata.identifiers(json.loads(schema)))
        removed = json.loads(
            db.execute(
                "SELECT removed_json FROM bt_release_plans WHERE id=?", (plan["id"],)
            ).fetchone()[0]
        )
        for item in plan["files"]:
            name = Path(item["path"]).stem
            if (
                item["path"].startswith("tables/")
                and name in set(retained_tables) | indexed_tables
            ):
                continue
            if item["path"].startswith("indexes/") and name in retained_blocks:
                continue
            path = _safe(store, item["path"])
            if path.exists():
                if not path.is_file() or _file_hash(path) != item["sha256"]:
                    raise RuntimeError(
                        "BT release file differs from registered authority"
                    )
                path.unlink()
            removed[item["path"]] = item["bytes"]
            db.execute(
                "UPDATE bt_release_plans SET removed_json=? WHERE id=?",
                (_json(removed), plan["id"]),
            )
    with store._connect() as db:
        db.execute("UPDATE bt_release_plans SET completed=1 WHERE id=?", (plan["id"],))
        removed = json.loads(
            db.execute(
                "SELECT removed_json FROM bt_release_plans WHERE id=?", (plan["id"],)
            ).fetchone()[0]
        )
    return {"id": plan["id"], "bytes": sum(removed.values())}
