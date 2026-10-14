from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt import BTStore


def fixture(tmp_path):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    frame = pl.DataFrame(
        {
            "time": [date(2024, 1, 1) + timedelta(days=i) for i in range(5)],
            "value": [float(i) for i in range(5)],
        }
    )
    return store, frame


def test_finite_memo_reuses_decodes_but_never_exposes_mutable_cached_frames(
    tmp_path, monkeypatch
):
    store, frame = fixture(tmp_path)
    reference = store.publish_table(frame)
    calls = []
    read = pl.read_parquet
    monkeypatch.setattr(
        pl, "read_parquet", lambda *a, **kw: (calls.append(a[0]), read(*a, **kw))[1]
    )
    with store.read_context(cache_bytes=frame.estimated_size() * 3) as reader:
        first = reader.read_reference(reference)
        first.drop_in_place("value")
        assert_frame_equal(reader.read_reference(reference), frame)
        selected = reader.read_reference(
            reference,
            filters=[{"column": "time", "operation": "ge", "value": date(2024, 1, 4)}],
        )
        assert_frame_equal(selected, frame.tail(2))
        assert len(calls) == 1
    assert reader._frame_cache is None
    assert_frame_equal(store.read_reference(reference), frame)
    assert len(calls) == 2


def test_memo_honors_byte_cap_eviction_disable_and_file_disappearance(
    tmp_path, monkeypatch
):
    store, frame = fixture(tmp_path)
    references = [
        store.publish_table(
            frame.with_columns((pl.col("value") + offset).alias("value"))
        )
        for offset in (0, 10)
    ]
    calls = []
    read = pl.read_parquet
    monkeypatch.setattr(
        pl, "read_parquet", lambda *a, **kw: (calls.append(a[0]), read(*a, **kw))[1]
    )
    with store.read_context(cache_bytes=frame.estimated_size()) as reader:
        reader.read_reference(references[0])
        reader.read_reference(references[1])
        reader.read_reference(references[0])
        assert len(calls) == 3
        assert reader._frame_cache.bytes <= frame.estimated_size()
        store._table_path(references[0]["table_id"]).unlink()
        with pytest.raises(ValueError, match="missing"):
            reader.read_reference(references[0])
    with store.read_context(cache_bytes=0) as reader:
        reader.read_reference(references[1])
        reader.read_reference(references[1])
        assert len(calls) == 5


def test_selected_composition_ignores_missing_unconsumed_leaves(tmp_path):
    store, frame = fixture(tmp_path)
    first, second = (
        store.publish_table(frame.head(2)),
        store.publish_table(frame.tail(3)),
    )
    ref = store.compose_reference([{"reference": first}, {"reference": second}])
    store._table_path(first["table_id"]).unlink()
    with store.read_context() as reader:
        assert_frame_equal(
            reader.read_reference(
                ref,
                filters=[
                    {"column": "time", "operation": "ge", "value": date(2024, 1, 3)}
                ],
            ),
            frame.tail(3),
        )


def test_delta_repair_composes_unaffected_suffix_without_a_numerical_read(
    tmp_path, monkeypatch
):
    store, frame = fixture(tmp_path)
    ref = store.publish_table(frame)
    delta = frame.slice(2, 1).with_columns(pl.lit(100.0).alias("value"))
    read = BTStore.read_reference
    monkeypatch.setattr(
        BTStore,
        "read_reference",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("repair decoded old values")
        ),
    )
    refs = store.publish_continuation(
        {"daily": delta},
        previous={"daily": ref},
        output_start=date(2024, 1, 3),
        output_end=date(2024, 1, 3),
    )
    monkeypatch.setattr(BTStore, "read_reference", read)
    assert_frame_equal(
        store.read_reference(refs["daily"]),
        pl.concat([frame.head(2), delta, frame.tail(2)]),
    )
