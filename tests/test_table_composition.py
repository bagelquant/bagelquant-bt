import json
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt import BTStore
from bagelquant_bt.store import _identity


def test_exact_key_and_checkpoint_selections_prune_original_leaves(
    tmp_path, monkeypatch
):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    day = date(2024, 1, 1)
    originals = [
        store.publish_table(
            pl.DataFrame(
                {
                    "checkpoint_date": [day + timedelta(days=index)],
                    "available_date": [day + timedelta(days=index)],
                    "asset_id": ["a" if index % 2 else "b"],
                    "value": [float(index)],
                }
            )
        )
        for index in range(4)
    ]
    ref = store.compose_reference([{"reference": item} for item in originals])
    reads = []
    scan = pl.scan_parquet

    def observed(path, *args, **kwargs):
        reads.append(str(path))
        return scan(path, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", observed)
    selected = store.read_reference(
        ref,
        filters=[
            {
                "column": "checkpoint_date",
                "operation": "eq",
                "value": day,
            }
        ],
    )
    assert selected["value"].to_list() == [0.0]
    assert len(reads) == 1
    reads.clear()
    keys = pl.DataFrame(
        {
            "asset_id": ["a", "b"],
            "available_date": [day + timedelta(days=1), day + timedelta(days=2)],
        }
    )
    selected = store.read_reference(ref, coordinates=keys)
    assert selected["value"].to_list() == [1.0, 2.0]
    assert len(reads) == 2
    reads.clear()
    assert store.read_reference(ref, coordinates=keys.head(0)).is_empty()
    assert not reads
    mismatched = keys.with_columns(pl.col("available_date").cast(pl.String))
    with pytest.raises(ValueError, match="original indexed nonnull types"):
        store.read_reference(ref, coordinates=mismatched)


def test_append_and_repair_references_preserve_original_bytes(tmp_path, monkeypatch):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(4)]
    frame = pl.DataFrame(
        {"time": days[:3], "window_id": ["1"] * 3, "value": [1.0, 2.0, 3.0]}
    )
    old = store.publish_shared("alpha", "signal", "first", {"daily": frame})
    old_ref = old["files"]["daily"]
    old_bytes = store._table_path(old_ref["table_id"]).read_bytes()
    delta = pl.DataFrame(
        {"time": days[2:], "window_id": ["1"] * 2, "value": [33.0, 4.0]}
    )
    tail = store.publish_table(delta)
    with monkeypatch.context() as patch:
        patch.setattr(
            pl,
            "read_parquet",
            lambda *_a, **_k: pytest.fail("composition decoded originals"),
        )
        composed = store.compose_reference(
            [
                {
                    "reference": old_ref,
                    "filters": [
                        {"column": "time", "operation": "le", "value": days[1]}
                    ],
                },
                {"reference": tail},
            ]
        )
        current = store.publish_shared(
            "alpha",
            "signal",
            "second",
            {},
            expected_shared=old,
            file_references={"daily": composed},
        )
    assert composed["rows"] == 4
    assert store._table_path(old_ref["table_id"]).read_bytes() == old_bytes
    assert_frame_equal(
        store.read_reference(current["files"]["daily"]),
        pl.concat([frame.head(2), delta]),
    )
    assert store.verify()["status"] == "complete"
    assert tail["table_id"] in {ref["table_id"] for ref in store.inventory()["tables"]}
    assert not store.cleanup_plan()["candidates"]
    additional = store.publish_table(delta.tail(1))
    again = store.compose_reference(
        [{"reference": composed}, {"reference": additional}]
    )
    assert len(again["parts"]) == 3
    assert all(part["reference"].get("format") is None for part in again["parts"])


def test_coordinate_index_audit_and_schema_selection_rejections(tmp_path):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frame = pl.DataFrame({"time": [date(2024, 1, 1)], "value": [1.0]})
    ref = store.publish_table(frame)
    with pytest.raises(ValueError, match="indexed coordinates"):
        store.compose_reference(
            [
                {
                    "reference": ref,
                    "filters": [{"column": "value", "operation": "eq", "value": 1}],
                }
            ]
        )
    other = store.publish_table(frame.with_columns(pl.col("value").cast(pl.Int64)))
    with pytest.raises(ValueError, match="schemas differ"):
        store.compose_reference([{"reference": ref}, {"reference": other}])
    with pytest.raises(ValueError, match="ISO date"):
        store.compose_reference(
            [
                {
                    "reference": ref,
                    "filters": [
                        {"column": "time", "operation": "eq", "value": "2024-99-99"}
                    ],
                }
            ]
        )
    for day in ("20240101", "2024-W01-1"):
        normalized = store.compose_reference(
            [
                {
                    "reference": ref,
                    "filters": [{"column": "time", "operation": "eq", "value": day}],
                }
            ]
        )
        assert normalized["rows"] == 1
        assert_frame_equal(store.read_reference(normalized), frame)
    with store._connect() as connection:
        index = json.loads(
            connection.execute(
                "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                (ref["table_id"],),
            ).fetchone()[0]
        )
        index["coordinates"] = store._metadata.save(
            pl.DataFrame(
                {"time": [date(2024, 2, 1)], "_rows": pl.Series([1], dtype=pl.UInt32)}
            )
        )
        index["digest"] = _identity(
            {
                "version": "bt.table.index.v3",
                "reference": ref,
                "schema": index["schema"],
                "columns": index["columns"],
                "coordinates": index["coordinates"],
            }
        )
        connection.execute(
            "UPDATE bt_table_index SET schema_json=? WHERE table_id=?",
            (json.dumps(index), ref["table_id"]),
        )
    with pytest.raises(ValueError, match="differs from original table rows"):
        store.verify_reference(ref, deep=True)


def test_original_column_order_requires_explicit_legacy_index_maintenance(tmp_path):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frame = pl.DataFrame({"time": [date(2024, 1, 1)], "z": [1.0], "a": [2.0]})
    shared = store.publish_shared("original", "primitive", "first", {"daily": frame})
    reference = shared["files"]["daily"]
    with store._connect() as connection:
        index = json.loads(
            connection.execute(
                "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                (reference["table_id"],),
            ).fetchone()[0]
        )
        index.pop("columns")
        index["digest"] = _identity(
            {
                "version": "bt.table.index.v3",
                "reference": reference,
                "schema": index["schema"],
                "coordinates": index["coordinates"],
            }
        )
        connection.execute(
            "UPDATE bt_table_index SET schema_json=? WHERE table_id=?",
            (json.dumps(index), reference["table_id"]),
        )
    with pytest.raises(ValueError, match="original-column index maintenance"):
        store.compose_reference([{"reference": reference}])
    assert store.maintain_reference(reference)["status"] == "complete"
    composed = store.compose_reference(
        [
            {
                "reference": reference,
                "filters": [
                    {"column": "time", "operation": "eq", "value": date(2024, 2, 1)}
                ],
            }
        ]
    )
    assert_frame_equal(store.read_reference(composed), frame.head(0))


@pytest.mark.parametrize("columns", [["a", "z", "time"], ["time", "z", "z"], "time"])
def test_original_column_index_audit_rejects_self_consistent_tampering(
    tmp_path, columns
):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frame = pl.DataFrame({"time": [date(2024, 1, 1)], "z": [1.0], "a": [2.0]})
    reference = store.publish_table(frame)
    with store._connect() as connection:
        index = json.loads(
            connection.execute(
                "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                (reference["table_id"],),
            ).fetchone()[0]
        )
        index["columns"] = columns
        index["digest"] = _identity(
            {
                "version": "bt.table.index.v3",
                "reference": reference,
                "schema": index["schema"],
                "columns": columns,
                "coordinates": index["coordinates"],
            }
        )
        connection.execute(
            "UPDATE bt_table_index SET schema_json=? WHERE table_id=?",
            (json.dumps(index), reference["table_id"]),
        )
    with pytest.raises(ValueError, match="original-column index"):
        store.verify_reference(reference, deep=True)


def test_sparse_date_coordinates_use_registered_types_after_hundred_nulls(
    tmp_path, monkeypatch
):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    first = date(2024, 1, 1)
    days = [first + timedelta(days=index) for index in range(151)]
    frame = pl.DataFrame(
        {
            "time": days,
            "available_date": [None] * 150 + [days[-1]],
            "asset_id": ["a"] * 151,
            "value": [1.0] * 151,
        },
        schema_overrides={"available_date": pl.Date},
    )
    reference = store.publish_table(frame)
    composed = store.compose_reference(
        [
            {
                "reference": reference,
                "filters": [{"column": "time", "operation": "le", "value": days[-1]}],
            }
        ]
    )
    keys = pl.DataFrame({"asset_id": ["a"], "available_date": [days[-1]]})
    assert_frame_equal(store.read_reference(composed, coordinates=keys), frame.tail(1))
    empty = store.publish_table(frame.head(150))
    scan = pl.scan_parquet
    reads = []
    monkeypatch.setattr(
        pl,
        "scan_parquet",
        lambda *args, **kwargs: (reads.append(args), scan(*args, **kwargs))[1],
    )
    assert store.read_reference(empty, coordinates=keys).is_empty()
    assert not reads


def test_cleanup_republication_and_final_reference_fence(tmp_path, monkeypatch):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frame = pl.DataFrame(
        {"time": [date(2024, 1, 1)], "window_id": ["1"], "value": [1.0]}
    )
    first = store.publish_table(frame)
    store.cleanup(store.cleanup_plan())
    second = store.publish_table(frame)
    assert store._table_path(second["table_id"]).is_file()
    assert second["logical_hash"] == first["logical_hash"]
    plan = store.cleanup_plan()
    original = store._publication_files

    def interrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        store.cleanup(plan)
        return result

    monkeypatch.setattr(store, "_publication_files", interrupted)
    with pytest.raises(ValueError, match="artifact is missing"):
        store.publish_shared(
            "alpha", "signal", "first", {}, file_references={"daily": second}
        )
    assert store.shared_receipt("alpha", "signal") is None


def test_continuation_accepts_window_id_without_window_kind(tmp_path):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frame = pl.DataFrame(
        {"time": [date(2024, 1, 1)], "window_id": ["1"], "value": [1.0]}
    )
    ref = store.publish_table(frame)
    current = pl.concat(
        [frame, frame.with_columns(pl.lit(date(2024, 1, 2)).alias("time"))]
    )
    refs = store.publish_continuation({"daily": current}, previous={"daily": ref})
    assert_frame_equal(store.read_reference(refs["daily"]), current)


def test_evaluation_continuation_writes_only_affected_tail_and_preserves_table_order(
    tmp_path, monkeypatch
):
    from bagelquant_bt import evaluate_alpha
    from test_incremental_alpha import SETTINGS, inputs

    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    node, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        node,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        **SETTINGS,
    )
    receipt = store.publish_shared(
        "alpha", "signal", "first", previous.tables, metadata={"state": previous.state}
    )
    original_bytes = {
        name: store._table_path(ref["table_id"]).read_bytes()
        for name, ref in receipt["files"].items()
    }
    node, labels, calendar, proof = inputs(36)
    current = evaluate_alpha(
        node,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        previous=previous,
        **SETTINGS,
    )
    writes = []
    publish = store.publish_table

    def tracked(frame, **kwargs):
        writes.append(frame)
        return publish(frame, **kwargs)

    monkeypatch.setattr(store, "publish_table", tracked)
    references = store.publish_continuation(current.tables, previous=receipt["files"])
    for name, frame in current.tables.items():
        assert_frame_equal(store.read_reference(references[name]), frame)
        assert (
            store._table_path(receipt["files"][name]["table_id"]).read_bytes()
            == original_bytes[name]
        )
    daily = [
        frame
        for frame in writes
        if "evaluation_date" in frame.columns or "time" in frame.columns
    ]
    assert daily and all(
        frame.height == 0
        or frame[
            next(key for key in ("evaluation_date", "time") if key in frame.columns)
        ].min()
        >= date(2024, 1, 30)
        for frame in daily
    )
    # No numerical writes for a complete unchanged result.
    writes.clear()
    assert store.publish_continuation(current.tables, previous=references) == references
    assert not writes
