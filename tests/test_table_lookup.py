"""Original receipt lookup preserves authority without walking unrelated trees."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date
from threading import Barrier

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


@pytest.mark.parametrize("kind", ["shared", "chapter"])
def test_publication_searches_history_once_for_all_missing_tables(
    store, monkeypatch, kind
):
    store.publish_shared("old", "signal", "old", {"rows": frame()})
    evaluation = store.begin("new", settings={}, through="2024-01-02")
    original = store._read_connection
    searches = []

    @contextmanager
    def monitored():
        with original() as connection:
            connection.set_trace_callback(
                lambda query: (
                    searches.append(query)
                    if query.startswith("SELECT manifest_json")
                    and ("FROM bt_shared" in query or "FROM bt_chapters" in query)
                    else None
                )
            )
            yield connection

    monkeypatch.setattr(store, "_read_connection", monitored)
    frames = {f"rows_{index}": frame(10.0 + index) for index in range(12)}
    if kind == "shared":
        receipt = store.publish_shared("new", "signal", "new", frames)
    else:
        receipt = store.publish_chapter(evaluation, "summary", frames)
    assert len(searches) <= 2, searches
    assert len(receipt["files"]) == len(frames)
    for name, reference in receipt["files"].items():
        assert store.read_reference(reference).equals(frames[name])


def test_batch_hashes_each_supplied_frame_once_and_reuses_aliases(store, monkeypatch):
    import bagelquant_bt.store as storage

    supplied = {"first": frame(), "alias": frame(), "other": frame(2.0)}
    hashes = []
    original = storage.hash_dataframe

    def hashed(value):
        hashes.append(id(value))
        return original(value)

    def unexpected_read(*_args, **_kwargs):
        pytest.fail("A duplicate new table must reuse its just-registered descriptor")

    monkeypatch.setattr(storage, "hash_dataframe", hashed)
    monkeypatch.setattr(pl, "read_parquet", unexpected_read)
    receipt = store.publish_shared("root", "signal", "new", supplied)
    assert hashes == [id(value) for value in supplied.values()]
    assert receipt["files"]["first"] == receipt["files"]["alias"]


@pytest.mark.parametrize("escaped", [0, 1, 64])
def test_batch_resolves_supplied_references_and_new_frames_together(
    store, monkeypatch, escaped
):
    original = store.publish_shared("old", "signal", "old", {"rows": frame()})
    reference = original["files"]["rows"]
    with store._connect() as connection:
        raw = connection.execute("SELECT manifest_json FROM bt_shared").fetchone()[0]
        identifier = reference["table_id"]
        replacement = (
            "".join(f"\\u{ord(char):04x}" for char in identifier[:escaped])
            + identifier[escaped:]
        )
        connection.execute(
            "UPDATE bt_shared SET manifest_json=?",
            (raw.replace(identifier, replacement),),
        )
        connection.execute("DELETE FROM bt_table_index")
    read = store._read_connection
    searches = []

    @contextmanager
    def monitored():
        with read() as connection:
            connection.set_trace_callback(
                lambda query: (
                    searches.append(query)
                    if query.startswith("SELECT manifest_json")
                    and ("FROM bt_shared" in query or "FROM bt_chapters" in query)
                    else None
                )
            )
            yield connection

    monkeypatch.setattr(store, "_read_connection", monitored)
    supplied = {"also_rows": frame(), **{f"new_{n}": frame(n + 10.0) for n in range(8)}}
    receipt = store.publish_shared(
        "new",
        "signal",
        "new",
        supplied,
        file_references={"rows": reference, "also_rows": reference},
    )
    assert len(searches) == 2
    assert receipt["files"]["rows"] == receipt["files"]["also_rows"] == reference


@pytest.mark.parametrize("kind", ["shared", "chapter"])
def test_batch_does_not_recreate_a_missing_unindexed_original(store, kind):
    original = store.publish_shared("old", "signal", "old", {"rows": frame()})
    reference = original["files"]["rows"]
    store._table_path(reference["table_id"]).unlink()
    with store._connect() as connection:
        connection.execute("DELETE FROM bt_table_index")
    evaluation = store.begin("new", settings={}, through="2024-01-02")
    supplied = {"new": frame(2.0), "missing": frame()}
    with pytest.raises(ValueError, match="artifact is missing"):
        if kind == "shared":
            store.publish_shared("new", "signal", "new", supplied)
        else:
            store.publish_chapter(evaluation, "summary", supplied)
    assert not store._table_path(reference["table_id"]).exists()
    assert store.shared_receipt("old", "signal") == original
    assert store.shared_receipt("new", "signal") is None
    assert store.chapter(evaluation, "summary")["status"] == "missing"


def test_batch_authenticates_matches_but_not_hashes_in_unrelated_metadata(store):
    import bagelquant_bt.store as storage

    unrelated = storage.hash_dataframe(frame(2.0))
    original = store.publish_shared(
        "old", "signal", "old", {"rows": frame()}, metadata={"unrelated": unrelated}
    )
    with store._connect() as connection:
        connection.execute("DELETE FROM bt_table_index")
        connection.execute(
            "UPDATE bt_shared SET manifest_json=?",
            (json.dumps({**original, "receipt_identity": "0" * 64}),),
        )
    # A hash outside files is not original authority, even in a damaged receipt.
    store.publish_shared("unrelated", "signal", "new", {"rows": frame(2.0)})
    with pytest.raises(ValueError, match="identity"):
        store.publish_shared(
            "new", "signal", "new", {"rows": frame(), "new": frame(3.0)}
        )


def test_batch_history_scan_checks_cancellation_before_writing(store, monkeypatch):
    original = store.publish_shared("old", "signal", "old", {"rows": frame()})
    read = store._read_connection
    scanning = [False]
    failure = RuntimeError("cancel original lookup")

    @contextmanager
    def monitored():
        with read() as connection:
            connection.set_trace_callback(
                lambda query: (
                    scanning.__setitem__(0, True)
                    if "SELECT manifest_json FROM bt_shared" in query
                    else None
                )
            )
            yield connection

    def canceled():
        if scanning[0]:
            raise failure

    monkeypatch.setattr(store, "_read_connection", monitored)
    before = set((store.artifact_path / "tables").iterdir())
    with pytest.raises(RuntimeError) as caught:
        store.publish_shared(
            "new", "signal", "new", {"rows": frame(2.0)}, check_canceled=canceled
        )
    assert caught.value is failure
    assert set((store.artifact_path / "tables").iterdir()) == before
    assert store.shared_receipt("old", "signal") == original
    assert store.shared_receipt("new", "signal") is None
    # Failed operation-local absence results cannot leak into a later retry.
    assert store.publish_shared("new", "signal", "new", {"rows": frame(2.0)})


def test_concurrent_identical_batches_keep_one_immutable_publication(
    store, monkeypatch
):
    barrier = Barrier(2)
    resolve = store._publication_table_references

    def together(*args):
        result = resolve(*args)
        barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(store, "_publication_table_references", together)

    def publish(_index):
        return store.publish_shared(
            "root", "signal", "same", {"first": frame(), "second": frame(2.0)}
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(publish, range(2)))
    assert first == second == store.shared_receipt("root", "signal")
    assert store.verify()["status"] == "complete"


@pytest.mark.parametrize("missing_index", [False, True])
def test_prepared_miss_does_not_replace_an_original_published_during_the_batch(
    store, monkeypatch, missing_index
):
    other = BTStore(store.meta_path, store.artifact_path)
    save = store._save_table
    published = []

    def interleaved(value, check_canceled, **kwargs):
        result = save(value, check_canceled, **kwargs)
        if not published:
            receipt = other.publish_shared(
                "other", "signal", "original", {"rows": frame(2.0)}
            )
            published.append(receipt)
            identifier = receipt["files"]["rows"]["table_id"]
            if missing_index:
                with other._connect() as connection:
                    connection.execute(
                        "DELETE FROM bt_table_index WHERE table_id=?", (identifier,)
                    )
            other._table_path(identifier).unlink()
        return result

    monkeypatch.setattr(store, "_save_table", interleaved)
    with pytest.raises(ValueError, match="artifact is missing"):
        store.publish_shared(
            "root", "signal", "new", {"first": frame(), "later": frame(2.0)}
        )
    assert published == [store.shared_receipt("other", "signal")]
    identifier = published[0]["files"]["rows"]["table_id"]
    assert not store._table_path(identifier).exists()
    assert store.shared_receipt("root", "signal") is None


@pytest.mark.parametrize("failure_kind", ["validation", "cancellation"])
def test_original_lookup_failure_releases_only_its_owned_large_aliases(
    store, monkeypatch, failure_kind
):
    import bagelquant_bt.store as storage

    receipt = store.publish_shared(
        "old",
        "signal",
        "old",
        {"rows": frame()},
        metadata={"padding": "x" * (2 * 1024 * 1024)},
    )
    reference = receipt["files"]["rows"]
    with store._connect() as connection:
        connection.execute("DELETE FROM bt_table_index")
    failure = KeyboardInterrupt()
    foreign = {"unchanged": object()}

    def interrupted(*_args, **_kwargs):
        assert foreign["unchanged"] is retained
        raise failure

    retained = foreign["unchanged"]
    if failure_kind == "validation":
        monkeypatch.setattr(storage, "_verify_record", interrupted)

        def invoke():
            return store.describe_reference(reference)
    else:
        # Cancellation during traversal follows successful decoding of a large
        # candidate; the helper must discard its own iterator and receipt aliases.
        calls = [0]

        def canceled():
            calls[0] += 1
            if calls[0] > 3:
                interrupted()

        def invoke():
            with store._read_connection() as connection:
                return store._original_table_references(
                    connection, {reference["table_id"]}, canceled
                )

    with pytest.raises(KeyboardInterrupt) as caught:
        invoke()
    assert caught.value is failure
    checked = False
    trace = failure.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code.co_name == "_original_table_references":
            local = trace.tb_frame.f_locals
            assert all(
                local[name] is None
                for name in ("raw", "row", "receipt", "matched", "value")
            )
            assert local["pending"] == [] and local["references"] == {}
            checked = True
        trace = trace.tb_next
    assert checked
    assert foreign["unchanged"] is retained
