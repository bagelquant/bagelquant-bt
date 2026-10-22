"""Original receipt lookup preserves authority without walking unrelated trees."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date

import polars as pl
import pytest

from bagelquant_bt import BTStore


@pytest.fixture
def store(tmp_path):
    result = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    result.initialize()
    return result


def frame(value=1.0):
    return pl.DataFrame({"time": [date(2024, 1, 2)], "value": [value]})


def test_new_table_does_not_decode_unrelated_original_descriptors(store, monkeypatch):
    store.publish_shared(
        "old", "signal", "original", {f"table_{index}": frame() for index in range(128)}
    )
    original = store._read_connection
    decoded = [0]
    with sqlite3.connect(":memory:") as decoder:

        def extract(value, path):
            decoded[0] += 1
            return decoder.execute(
                "SELECT json_extract(?,?)", (value, path)
            ).fetchone()[0]

        @contextmanager
        def monitored():
            with original() as connection:
                connection.create_function("json_extract", 2, extract)
                yield connection

        monkeypatch.setattr(store, "_read_connection", monitored)
        reference = store.publish_table(frame(2.0))
    assert decoded[0] == 0
    assert reference["rows"] == 1


@pytest.mark.parametrize("kind", ["shared", "chapter"])
@pytest.mark.parametrize("escaped", [0, 1, 64])
def test_missing_index_resolves_literal_and_unicode_escaped_originals(
    store, kind, escaped
):
    if kind == "shared":
        receipt = store.publish_shared("root", "signal", "original", {"rows": frame()})
        table = "bt_shared"
    else:
        evaluation = store.begin("root", settings={}, through="2024-01-02")
        receipt = store.publish_chapter(evaluation, "summary", {"rows": frame()})
        table = "bt_chapters"
    reference = receipt["files"]["rows"]
    identifier = reference["table_id"]
    with store._connect() as connection:
        raw = connection.execute(f"SELECT manifest_json FROM {table}").fetchone()[0]
        replacement = (
            "".join(f"\\u{ord(char):04x}" for char in identifier[:escaped])
            + identifier[escaped:]
        )
        changed = raw.replace(identifier, replacement)
        assert json.loads(changed) == json.loads(raw)
        connection.execute(f"UPDATE {table} SET manifest_json=?", (changed,))
        connection.execute("DELETE FROM bt_table_index")
    assert store.describe_reference(reference) == reference
    store._table_path(identifier).unlink()
    with pytest.raises(ValueError, match="artifact is missing"):
        store.publish_table(frame())


def test_missing_index_resolves_an_original_nested_in_a_composition(store):
    original = store.publish_table(frame())
    delta = pl.DataFrame({"time": [date(2024, 1, 3)], "value": [2.0]})
    references = store.publish_continuation(
        {"rows": delta},
        previous={"rows": original},
        output_start=date(2024, 1, 3),
        output_end=date(2024, 1, 3),
    )
    store.publish_shared("root", "signal", "composed", {}, file_references=references)
    with store._connect() as connection:
        connection.execute("DELETE FROM bt_table_index")
    assert store.describe_reference(original) == original


def test_hash_in_unrelated_metadata_cannot_authorize_a_reference(store):
    missing = "f" * 64
    store.publish_shared(
        "root", "signal", "original", {"rows": frame()}, metadata={"unrelated": missing}
    )
    with pytest.raises(ValueError, match="differs from its registered record"):
        store.describe_reference(
            {
                "backend": "bt",
                "table_id": missing,
                "logical_hash": missing,
                "checksum": missing,
                "rows": 0,
            }
        )


def test_matching_original_still_authenticates_its_receipt(store):
    receipt = store.publish_shared("root", "signal", "original", {"rows": frame()})
    with store._connect() as connection:
        damaged = {**receipt, "receipt_identity": "0" * 64}
        connection.execute(
            "UPDATE bt_shared SET manifest_json=?", (json.dumps(damaged),)
        )
        connection.execute("DELETE FROM bt_table_index")
    with pytest.raises(ValueError, match="identity"):
        store.describe_reference(receipt["files"]["rows"])
