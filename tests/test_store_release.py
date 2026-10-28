"""Release only selected evaluation files and preserve shared original tables."""

import pytest

from bagelquant_bt.store import BTStore
from test_store import _frame


def test_release_shared_files_audit_and_repeat_publication(tmp_path):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    first = store.begin("one", settings={}, through="2024-01-03")
    second = store.begin("two", settings={}, through="2024-01-03")
    original = store.publish_chapter(
        first, "summary", {"shared": _frame(), "unique": _frame((5.0, 6.0))}
    )
    store.publish_chapter(second, "summary", {"rows": _frame()})
    plan = store.plan_release([first], owner_ids=["one"])
    assert store.apply_release(plan) == store.apply_release(plan)
    assert store.describe(first) is None
    assert store.chapter(first, "summary")["status"] == "missing"
    assert store.verify()["status"] == "complete"
    assert len(list((store.artifact_path / "tables").glob("*.parquet"))) == 1
    with store._read_connection() as db:
        assert db.execute("SELECT count(*) FROM bt_release_plans").fetchone()[0] == 1
    assert store.begin("one", settings={}, through="2024-01-03") == first
    restored = store.publish_chapter(
        first, "summary", {"shared": _frame(), "unique": _frame((5.0, 6.0))}
    )
    assert restored["files"] == original["files"]
    assert store.verify()["status"] == "complete"


def test_release_stale_plan(tmp_path):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    first = store.begin("one", settings={}, through="2024-01-03")
    store.publish_chapter(first, "summary", {"rows": _frame()})
    plan = store.plan_release([first])
    store.begin("two", settings={}, through="2024-01-03")
    with pytest.raises(RuntimeError, match="changed"):
        store.apply_release(plan)
    assert store.describe(first)


def test_release_without_optional_table_index(tmp_path):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    evaluation = store.begin("one", settings={}, through="2024-01-03")
    store.publish_chapter(evaluation, "summary", {"rows": _frame()})
    with store._connect() as db:
        db.execute("DROP TABLE bt_table_index")
    plan = store.plan_release([evaluation])
    store.apply_release(plan)
    assert store.release_status(plan["id"]) == "completed"
    assert not list((store.artifact_path / "tables").glob("*.parquet"))
    assert store.verify()["status"] == "complete"


def test_release_refuses_corrupt_retained_receipt(tmp_path):
    import json

    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    first = store.begin("one", settings={}, through="2024-01-03")
    second = store.begin("two", settings={}, through="2024-01-03")
    store.publish_chapter(first, "summary", {"rows": _frame()})
    saved = store.publish_chapter(second, "summary", {"rows": _frame()})
    saved["files"] = {}
    with store._connect() as db:
        db.execute(
            "UPDATE bt_chapters SET manifest_json=? WHERE evaluation_id=?",
            (json.dumps(saved), second),
        )
    with pytest.raises(ValueError, match="identity"):
        store.plan_release([first])
    assert store.describe(first)
