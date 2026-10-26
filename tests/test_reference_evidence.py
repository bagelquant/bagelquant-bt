import json
from datetime import date, timedelta

import polars as pl

from bagelquant_bt import BTStore
from bagelquant_bt.table_composition import coordinates


def inputs():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(12)]
    frame = pl.DataFrame(
        [
            {
                "time": day,
                "available_date": days[index + 1],
                "factor": factor,
                "family": "industry" if factor == "new" else "style",
                "factor_return": index * 0.01,
            }
            for index, day in enumerate(days[:-1])
            for factor in ("size", "new")
            if factor != "new" or index >= 6
        ]
    )
    return days, frame


def test_interval_proof_and_catalog_are_metadata_only_and_fragmentation_invariant(
    tmp_path, monkeypatch
):
    days, frame = inputs()
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    full = store.publish_table(frame)
    parts = [
        store.publish_table(frame.filter(pl.col("factor") == name))
        for name in ("size", "new")
    ]
    split = store.compose_reference([{"reference": part} for part in parts])
    clipped = store.publish_table(frame.filter(pl.col("available_date") <= days[7]))
    monkeypatch.setattr(
        BTStore,
        "read_reference",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("metadata proof read values")
        ),
    )
    assert store.reference_evidence(full) == store.reference_evidence(split)
    evidence = store.reference_evidence(full)
    assert evidence["factors"]["new"] == {
        "family": "industry",
        "first_seen": str(days[6]),
    }
    assert store.reference_interval_identity(
        full, start=days[0], end=days[8], cutoff=days[7]
    ) == (store.reference_interval_identity(clipped, start=days[0], end=days[8]))
    changed = frame.with_columns(
        pl.when(pl.col("time") == days[10])
        .then(100.0)
        .otherwise(pl.col("factor_return"))
        .alias("factor_return")
    )
    new = store.publish_table(changed)
    assert store.reference_interval_identity(full, start=days[0], end=days[6]) == (
        store.reference_interval_identity(new, start=days[0], end=days[6])
    )
    assert store.reference_interval_identity(full, start=days[9], end=days[10]) != (
        store.reference_interval_identity(new, start=days[9], end=days[10])
    )


def test_legacy_evidence_requires_explicit_index_maintenance_preserving_receipts(
    tmp_path, monkeypatch
):
    days, frame = inputs()
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    receipt = store.publish_shared("risk", "returns", "original", {"returns": frame})
    ref = receipt["files"]["returns"]
    before = store._table_path(ref["table_id"]).read_bytes()
    with store._connect() as connection:
        connection.execute(
            "UPDATE bt_table_index SET schema_json=? WHERE table_id=?",
            (
                store._index_schema(ref, frame.schema, coordinates=coordinates(frame)),
                ref["table_id"],
            ),
        )
    assert store.reference_evidence(ref) is None
    assert store.reference_interval_identity(ref, start=days[0], end=days[7]) is None
    plan = store.index_plan(evidence=True, references=[ref])
    assert len(plan["tables"]) == 1
    assert store.build_index(plan)["status"] == "complete"
    assert store.index_plan(evidence=True, references=[ref])["tables"] == []
    assert store._table_path(ref["table_id"]).read_bytes() == before
    assert store.shared_receipt("risk", "returns", input_identity="original") == receipt
    assert store.reference_evidence(ref)["factors"]["size"]["first_seen"] == str(
        days[0]
    )
    with store._read_connection() as connection:
        index = json.loads(
            connection.execute("SELECT schema_json FROM bt_table_index").fetchone()[0]
        )
    assert index["evidence"]["groups"]["rows"] == 11


def test_filtered_daily_evidence_matches_consumed_maturity_without_value_reads(
    tmp_path, monkeypatch
):
    days, frame = inputs()
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    reference = store.publish_table(frame)
    clipped = store.publish_table(frame.filter(pl.col("available_date") <= days[7]))
    before = store.meta_path.read_bytes()
    monkeypatch.setattr(
        pl,
        "read_parquet",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("proof read values")),
    )
    monkeypatch.setattr(
        pl,
        "scan_parquet",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("proof scanned values")),
    )
    actual = store.reference_evidence(
        reference,
        filters=[{"column": "available_date", "operation": "le", "value": days[7]}],
    )
    assert actual["dates"] == store.reference_evidence(clipped)["dates"]
    assert sum(actual["date_rows"].values()) == clipped["rows"]
    later = store.reference_evidence(
        reference,
        filters=[{"column": "available_date", "operation": "le", "value": days[9]}],
    )
    assert all(later["dates"][day] == token for day, token in actual["dates"].items())
    assert store.meta_path.read_bytes() == before
