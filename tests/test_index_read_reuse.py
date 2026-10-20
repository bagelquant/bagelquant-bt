"""Call-local registered-index decoding; current metadata is never memoized."""

import json
import os
import struct
from contextlib import contextmanager
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import bagelquant_bt.store as store_module
from bagelquant_bt import BTStore


@pytest.fixture
def indexed(tmp_path):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frame = pl.DataFrame(
        {
            "time": [date(2024, 1, 1), date(2024, 1, 2)],
            "asset_id": ["甲", "b"],
            "value": [-0.0, None],
            "observed_count": pl.Series([1, 0], dtype=pl.UInt32),
        },
        schema_overrides={"value": pl.Float64},
    )
    saved = store.publish_shared("alpha", "signal", "original", {"daily": frame})
    reference = saved["files"]["daily"]
    with store._read_connection() as connection:
        row = tuple(connection.execute(
            "SELECT reference_json,schema_json FROM bt_table_index WHERE table_id=?",
            (reference["table_id"],),
        ).fetchone())
    return store, frame, reference, row


def assert_exact(actual, expected):
    assert_frame_equal(actual, expected, check_exact=True)
    assert actual.schema == expected.schema
    if "value" in expected.columns and expected.schema["value"] == pl.Float64:
        for left, right in zip(actual["value"], expected["value"], strict=True):
            if right is None:
                assert left is None
            else:
                assert struct.pack(">d", left) == struct.pack(">d", right)


def mutate_index(store, reference, row, change):
    """Only owner-test fixture metadata; every root is pytest's temporary root."""
    with store._connect() as connection:
        if change == "missing":
            connection.execute(
                "DELETE FROM bt_table_index WHERE table_id=?",
                (reference["table_id"],),
            )
        elif change == "table_missing":
            connection.execute("DROP TABLE bt_table_index")
        else:
            column, value = (
                ("schema_json", row[1] + " ")
                if change == "schema"
                else ("reference_json", row[0] + " ")
            )
            # Whitespace changes are logically valid, but no longer the exact
            # serialization validated in T1; they cannot use its parsed index.
            connection.execute(
                f"UPDATE bt_table_index SET {column}=? WHERE table_id=?",
                (value, reference["table_id"]),
            )


@pytest.mark.parametrize("selected", [False, True])
def test_one_index_decode_per_call_preserves_fresh_transactions_and_cache(
    indexed, monkeypatch, selected
):
    store, frame, reference, row = indexed
    original_loads = json.loads
    decoded, observed = [], []

    def counted(value, *args, **kwargs):
        if value == row[1]:
            decoded.append(value)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(store_module.json, "loads", counted)
    filters = ([{"column": "time", "operation": "eq", "value": frame["time"][0]}]
               if selected else [])
    expected = frame.head(1) if selected else frame
    with store.read_context() as reader:
        fresh = reader._fresh_reference_index

        def observe(registered):
            observed.append(registered)
            return fresh(registered)

        monkeypatch.setattr(reader, "_fresh_reference_index", observe)
        assert_exact(reader.read_reference(reference, filters=filters), expected)
        assert len(decoded) == 1 and len(observed) == 2  # T2 and physical T3.
        assert all(record == {} for record in observed)
        if selected:
            # Prove the already memoized selection needs no Parquet scan.
            monkeypatch.setattr(
                pl, "scan_parquet", lambda *_a, **_k: pytest.fail("cache miss")
            )
        else:
            monkeypatch.setattr(
                pl, "read_parquet", lambda *_a, **_k: pytest.fail("cache miss")
            )
        assert_exact(reader.read_reference(reference, filters=filters), expected)
        assert len(decoded) == 2 and len(observed) == 3  # Fresh T1/T2 on a hit.
        assert all(record == {} for record in observed)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("change", ["schema", "reference", "missing", "table_missing"])
def test_t2_fresh_row_change_rejects_even_a_populated_frame_cache(
    indexed, monkeypatch, cached, change
):
    store, frame, reference, row = indexed
    with store.read_context() as reader:
        if cached:
            assert_exact(reader.read_reference(reference), frame)
        fresh = reader._fresh_reference_index
        observed = []

        def race(registered):
            observed.append(registered)
            mutate_index(store, reference, row, change)
            return fresh(registered)

        monkeypatch.setattr(reader, "_fresh_reference_index", race)
        monkeypatch.setattr(
            pl, "read_parquet", lambda *_a, **_k: pytest.fail("read after T2 race")
        )
        with pytest.raises(ValueError, match="index changed during read"):
            reader.read_reference(reference)
        assert len(observed) == 1 and observed[0] == {}


@pytest.mark.parametrize("change", ["schema", "reference", "missing", "table_missing"])
def test_t3_fresh_row_change_rejects_after_physical_leaf_read(
    indexed, monkeypatch, change
):
    store, _, reference, row = indexed
    with store.read_context(cache_bytes=0) as reader:
        fresh = reader._fresh_reference_index
        calls, observed = [], []

        def race(registered):
            calls.append(1)
            observed.append(registered)
            if len(calls) == 2:
                mutate_index(store, reference, row, change)
            return fresh(registered)

        monkeypatch.setattr(reader, "_fresh_reference_index", race)
        with pytest.raises(ValueError, match="index changed during read"):
            reader.read_reference(reference)
        assert len(calls) == 2 and all(record == {} for record in observed)
        assert reader._frame_cache.bytes == 0


@pytest.mark.parametrize("corruption", ["digest", "binding", "columns"])
def test_t1_canonical_validation_is_not_bypassed_by_an_old_cached_frame(
    indexed, corruption
):
    store, frame, reference, row = indexed
    with store.read_context() as reader:
        assert_exact(reader.read_reference(reference), frame)
        descriptor, index = json.loads(row[0]), json.loads(row[1])
        if corruption == "digest":
            index["digest"] = "invalid"
        elif corruption == "binding":
            descriptor["logical_hash"] = "0" * 64
        else:
            index["columns"] = ["value", "value"]
        with store._connect() as connection:
            connection.execute(
                "UPDATE bt_table_index SET reference_json=?,schema_json=? "
                "WHERE table_id=?",
                (json.dumps(descriptor), json.dumps(index), reference["table_id"]),
            )
        with pytest.raises(ValueError, match="index"):
            reader.read_reference(reference)


def test_physical_typed_schema_check_is_retained(indexed, monkeypatch):
    store, frame, reference, _ = indexed
    monkeypatch.setattr(
        pl, "read_parquet",
        lambda *_a, **_k: frame.with_columns(pl.col("value").cast(pl.Float32)),
    )
    with pytest.raises(ValueError, match="typed schema differs"):
        store.read_reference(reference)


def test_file_stat_change_still_invalidates_frame_memo(indexed, monkeypatch):
    store, frame, reference, _ = indexed
    with store.read_context() as reader:
        assert_exact(reader.read_reference(reference), frame)
        path = store._table_path(reference["table_id"])
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        reads = []
        original = pl.read_parquet

        def observed(*args, **kwargs):
            reads.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(pl, "read_parquet", observed)
        assert_exact(reader.read_reference(reference), frame)
        assert reads == [1]


def test_metadata_is_fresh_again_on_the_next_call_not_a_cross_call_cache(indexed):
    store, frame, reference, row = indexed
    with store.read_context() as reader:
        assert_exact(reader.read_reference(reference), frame)
        mutate_index(store, reference, row, "schema")
        # A later call validates its own new (logically equal) serialization.
        assert_exact(reader.read_reference(reference), frame)
        assert reader.describe_reference(reference) == reference


def test_index_free_registered_history_retains_unfiltered_read_and_selected_error(
    indexed
):
    store, frame, reference, _ = indexed
    with store._connect() as connection:
        connection.execute("DROP TABLE bt_table_index")
    assert store.describe_reference(reference) == reference
    assert_exact(store.read_reference(reference), frame)
    with pytest.raises(ValueError, match="coordinate-index maintenance"):
        store.read_reference(reference, filters=[
            {"column": "time", "operation": "eq", "value": frame["time"][0]}
        ])


def test_index_appearance_after_fallback_lookup_fails_closed(indexed, monkeypatch):
    store, _, reference, row = indexed
    with store._connect() as connection:
        connection.execute("DROP TABLE bt_table_index")
    with store.read_context() as reader:
        fresh = reader._fresh_reference_index

        def appeared(registered):
            with store._connect() as connection:
                connection.execute(
                    "CREATE TABLE bt_table_index(table_id TEXT PRIMARY KEY,"
                    "reference_json TEXT NOT NULL,schema_json TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO bt_table_index VALUES(?,?,?)",
                    (reference["table_id"], *row),
                )
            return fresh(registered)

        monkeypatch.setattr(reader, "_fresh_reference_index", appeared)
        with pytest.raises(ValueError, match="index changed during read"):
            reader.read_reference(reference)


@pytest.mark.parametrize(
    "primary", [ValueError("read"), KeyboardInterrupt(), SystemExit(4)]
)
def test_call_local_record_and_own_aliases_clear_without_changing_primary_error(
    indexed, monkeypatch, primary
):
    store, _, reference, _ = indexed
    records = []
    with store.read_context(cache_bytes=0) as reader:
        fresh = reader._fresh_reference_index

        def capture(registered):
            records.append(registered)
            return fresh(registered)

        def fail(*_a, **_k):
            raise primary

        monkeypatch.setattr(reader, "_fresh_reference_index", capture)
        monkeypatch.setattr(pl, "read_parquet", fail)
        with pytest.raises(type(primary)) as raised:
            reader.read_reference(reference)
        assert raised.value is primary and records and all(r == {} for r in records)
        traceback = raised.value.__traceback__
        checked = 0
        while traceback is not None:
            if traceback.tb_frame.f_code.co_name == "_read_registered_reference":
                assert traceback.tb_frame.f_locals["index"] is None
                assert traceback.tb_frame.f_locals.get("facts") is None
                checked += 1
            traceback = traceback.tb_next
        assert checked == 1 and reader._frame_cache.bytes == 0
    assert reader._frame_cache is None and reader._verification is None


def test_t1_parse_failure_clears_only_own_record_and_keeps_exception_identity(
    indexed, monkeypatch
):
    store, _, reference, row = indexed
    original = json.loads
    primary = KeyboardInterrupt()

    def fail_index(value, *args, **kwargs):
        if value == row[1]:
            raise primary
        return original(value, *args, **kwargs)

    monkeypatch.setattr(store_module.json, "loads", fail_index)
    with pytest.raises(KeyboardInterrupt) as raised:
        store.read_reference(reference)
    assert raised.value is primary
    traceback = primary.__traceback__
    checked = 0
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == "_registered_table_read":
            local = traceback.tb_frame.f_locals
            assert local["registered"] == {}
            assert all(
                local[name] is None for name in ("row", "reference", "index", "receipt")
            )
            checked += 1
        traceback = traceback.tb_next
    assert checked == 1


def test_concat_selection_empty_schema_and_leaf_pruning_stay_exact(
    indexed, monkeypatch
):
    store, frame, reference, _ = indexed
    second = frame.with_columns(pl.col("time") + timedelta(days=2))
    tail = store.publish_table(second)
    composed = store.compose_reference([
        {"reference": reference}, {"reference": tail},
    ])
    scans = []
    original = pl.scan_parquet

    def observed(*args, **kwargs):
        scans.append(str(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", observed)
    selected = store.read_reference(composed, filters=[
        {"column": "time", "operation": "le", "value": frame["time"][-1]}
    ])
    assert_exact(selected, frame)
    assert len(scans) == 1
    empty = store.read_reference(
        composed, coordinates=frame.select("time", "asset_id").head(0)
    )
    assert empty.schema == frame.schema and empty.is_empty()
    assert len(scans) == 1


def test_append_and_finite_repair_use_original_refs_without_historical_producers(
    indexed, monkeypatch
):
    store, frame, reference, _ = indexed
    path = store._table_path(reference["table_id"])
    before = path.read_bytes(), path.stat().st_mtime_ns
    delta = frame.tail(1).with_columns(
        pl.lit(date(2024, 1, 3)).alias("time"), pl.lit(3.0).alias("value")
    )
    with monkeypatch.context() as patch:
        patch.setattr(pl, "read_parquet", lambda *_a, **_k: pytest.fail("old read"))
        patch.setattr(pl, "scan_parquet", lambda *_a, **_k: pytest.fail("old scan"))
        appended = store.publish_continuation(
            {"daily": delta}, previous={"daily": reference},
            output_start=date(2024, 1, 3),
        )["daily"]
    expected = pl.concat([frame, delta])
    assert_exact(store.read_reference(appended), expected)
    revision = frame.tail(1).with_columns(pl.lit(22.0).alias("value"))
    with monkeypatch.context() as patch:
        patch.setattr(pl, "read_parquet", lambda *_a, **_k: pytest.fail("old read"))
        patch.setattr(pl, "scan_parquet", lambda *_a, **_k: pytest.fail("old scan"))
        repaired = store.publish_continuation(
            {"daily": revision}, previous={"daily": appended},
            output_start=date(2024, 1, 2), output_end=date(2024, 1, 2),
        )["daily"]
    assert_exact(
        store.read_reference(repaired), pl.concat([frame.head(1), revision, delta])
    )
    assert store.read_reference(repaired).schema["observed_count"] == pl.UInt32
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert store.describe_reference(reference) == reference
    assert store.verify()["status"] == "complete"


@pytest.mark.parametrize("fail", [False, True])
def test_lookup_transactions_close_before_reader_body(indexed, monkeypatch, fail):
    store, _, reference, _ = indexed
    active = []
    original = store._read_connection

    @contextmanager
    def tracked():
        active.append(1)
        try:
            with original() as connection:
                yield connection
        finally:
            active.pop()

    monkeypatch.setattr(store, "_read_connection", tracked)
    with store._registered_table_read(reference["table_id"]) as registered:
        assert not active and registered["reference"] == reference
    assert registered == {} and not active
    with store.read_context() as reader:
        # read_context copies the store; no snapshot may span numerical work.
        def body(registered, **_kwargs):
            assert not active
            if fail:
                raise ValueError("body")
            return pl.DataFrame()

        monkeypatch.setattr(reader, "_read_registered_reference", body)
        if fail:
            with pytest.raises(ValueError, match="body"):
                reader.read_reference(reference)
        else:
            assert reader.read_reference(reference).is_empty()
        assert not active


def test_changed_index_cannot_hide_behind_empty_coordinate_selection(
    indexed, monkeypatch
):
    store, frame, reference, row = indexed
    with store.read_context() as reader:
        fresh = reader._fresh_reference_index

        def race(registered):
            mutate_index(store, reference, row, "schema")
            return fresh(registered)

        monkeypatch.setattr(reader, "_fresh_reference_index", race)
        with pytest.raises(ValueError, match="index changed during read"):
            reader.read_reference(
                reference, coordinates=frame.select("time", "asset_id").head(0)
            )


def test_final_artifact_stat_check_is_not_replaced_by_index_equality(
    indexed, monkeypatch
):
    store, _, reference, _ = indexed
    path = store._table_path(reference["table_id"])
    original = pl.read_parquet

    def changed(*args, **kwargs):
        frame = original(*args, **kwargs)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return frame

    monkeypatch.setattr(pl, "read_parquet", changed)
    with pytest.raises(ValueError, match="artifact changed during read"):
        store.read_reference(reference)


def test_nested_composition_flattens_and_revalidates_each_cached_leaf(
    indexed, monkeypatch
):
    store, frame, first, _ = indexed
    second_frame = frame.with_columns(pl.col("time") + timedelta(days=2))
    third_frame = frame.with_columns(pl.col("time") + timedelta(days=4))
    second = store.publish_table(second_frame)
    third = store.publish_table(third_frame)
    inner = store.compose_reference([
        {"reference": first}, {"reference": second},
    ])
    outer = store.compose_reference([
        {"reference": inner}, {"reference": third},
    ])
    assert len(outer["parts"]) == 3
    assert all(part["reference"].get("format") is None for part in outer["parts"])
    schemas = {}
    with store._read_connection() as connection:
        for reference in (outer, first, second, third):
            row = connection.execute(
                "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                (reference["table_id"],),
            ).fetchone()
            schemas[row[0]] = reference["table_id"]
    decoded = []
    original = json.loads

    def counted(value, *args, **kwargs):
        if value in schemas:
            decoded.append(schemas[value])
        return original(value, *args, **kwargs)

    monkeypatch.setattr(store_module.json, "loads", counted)
    with store.read_context() as reader:
        expected = pl.concat([frame, second_frame, third_frame])
        assert_exact(reader.read_reference(outer), expected)
        assert sorted(decoded) == sorted(schemas.values())
        decoded.clear()
        monkeypatch.setattr(
            pl, "read_parquet", lambda *_a, **_k: pytest.fail("cached leaf read")
        )
        assert_exact(reader.read_reference(outer), expected)
        assert sorted(decoded) == sorted(schemas.values())
