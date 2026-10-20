import json
from datetime import date, timedelta

import polars as pl
import pytest

from bagelquant_bt import BTStore, index_blocks


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(index_blocks, "MAX_ROWS", 2)
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frame = pl.DataFrame(
        {
            "time": [date(2024, 1, 1) + timedelta(days=i) for i in range(6)],
            "asset_id": ["A"] * 6,
            "value": list(range(6)),
        }
    )
    reference = store.publish_table(frame)
    with store._read_connection() as connection:
        header = json.loads(
            connection.execute(
                "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                (reference["table_id"],),
            ).fetchone()[0]
        )
    return store, frame, reference, header


def test_keys_prune_blocks_empty_keys_read_nothing_and_grids_are_shared(
    tmp_path, monkeypatch
):
    store, frame, reference, header = fixture(tmp_path, monkeypatch)
    other = store.publish_table(frame.with_columns(pl.col("value") + 10))
    with store._read_connection() as connection:
        second = json.loads(
            connection.execute(
                "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                (other["table_id"],),
            ).fetchone()[0]
        )
    assert header["coordinates"] == second["coordinates"]
    reads = []
    read = index_blocks.MetadataBlocks._read

    def observed(self, block, descriptor):
        reads.append(block["id"])
        return read(self, block, descriptor)

    monkeypatch.setattr(index_blocks.MetadataBlocks, "_read", observed)
    keys = frame.select("time").slice(2, 1)
    assert store.read_reference(reference, coordinates=keys)["value"].to_list() == [2]
    assert len(reads) == 1
    reads.clear()
    assert store.read_reference(reference, coordinates=keys.head(0)).is_empty()
    assert reads == []


def test_deep_audit_rejects_resealed_false_bounds(tmp_path, monkeypatch):
    store, _, reference, header = fixture(tmp_path, monkeypatch)
    for block in header["coordinates"]["blocks"]:
        block["bounds"]["time"].update(min="2030-01-01", max="2030-01-02")
    header.pop("digest")
    with store._connect() as connection:
        connection.execute(
            "UPDATE bt_table_index SET schema_json=?",
            (store._seal_index(reference, header),),
        )
    with pytest.raises(ValueError, match="pruning bounds"):
        store.verify_reference(reference, deep=True)


@pytest.mark.parametrize("memo", [False, True])
def test_consumed_metadata_is_fenced_through_value_or_memo_return(
    tmp_path, monkeypatch, memo
):
    store, frame, reference, header = fixture(tmp_path, monkeypatch)
    block = store._metadata.path(header["coordinates"]["blocks"][0]["id"])
    with store.read_context() as reader:
        if memo:
            reader.read_reference(reference)
        if memo:
            original = reader._frame_cache.get

            def changed(*args):
                result = original(*args)
                block.write_bytes(b"changed")
                return result

            monkeypatch.setattr(reader._frame_cache, "get", changed)
        else:
            original = pl.scan_parquet

            def changed(*args, **kwargs):
                block.write_bytes(b"changed")
                return original(*args, **kwargs)

            monkeypatch.setattr(pl, "scan_parquet", changed)
        with pytest.raises(ValueError, match="metadata block changed"):
            reader.read_reference(reference, coordinates=frame.select("time").head(1))


def test_maintenance_keeps_composed_references_and_cleanup_ignores_directories(
    tmp_path, monkeypatch
):
    store, frame, reference, _ = fixture(tmp_path, monkeypatch)
    other = store.publish_table(frame.with_columns(pl.col("value") + 10))
    composed = store.compose_reference([{"reference": reference}, {"reference": other}])
    store.maintain_reference(composed)
    with store._read_connection() as connection:
        header = json.loads(
            connection.execute(
                "SELECT schema_json FROM bt_table_index WHERE table_id=?",
                (composed["table_id"],),
            ).fetchone()[0]
        )
    assert header["coordinates"] == {"composition": True}
    unknown = store._metadata.root / ("a" * 64 + ".parquet")
    unknown.mkdir()
    store.cleanup_plan()
    assert unknown.is_dir()
