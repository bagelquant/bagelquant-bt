"""Explicit same-boundary reuse never manufactures causal continuation state."""

from copy import deepcopy
from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from bagelquant_bt import SharedInputEvidence
from bagelquant_bt.input_maintenance import CHANNELS
from bagelquant_bt.store import BTStore, _identity, evaluation_identity


def setup(tmp_path):
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    first, last = date(2024, 1, 2), date(2024, 1, 3)
    settings, kernel = {"annualization": 240}, "alpha.evaluation.v1"
    before = {
        "node": "same-node",
        "value": "1" * 64,
        "sources": {"price": "old"},
        "through": last,
    }
    after = {**before, "value": "2" * 64, "sources": {"price": "new"}}
    identity = evaluation_identity(
        "signal", inputs=before, settings=settings, kernel=kernel
    )
    frame = pl.DataFrame({"time": [first, last], "return": [0.0, 1.0]})
    original = store.publish_shared(
        "owner",
        "signal",
        identity,
        {"daily_book_returns": frame},
        metadata={"groups": ["book_tail"], "through": last.isoformat()},
        sources={"price": "old"},
    )
    consumed = {}
    for name in CHANNELS:
        common = dict(
            content_token=_identity(
                {
                    "channel": name,
                    "rows": [[str(first), "A", None], [str(last), "B", -0.0]],
                }
            ),
            start=first,
            end=last,
            source_information_cutoff=date(2024, 1, 8),
            schema={"time": "Date", "asset_id": "String", "value": "Float64"},
            selector={
                "view": "snapshot",
                "strict": False,
                "include_historical_baseline": False,
            },
            complete=True,
        )
        consumed[name] = {
            "original": SharedInputEvidence(
                source_receipt=before["value"]
                if name in {"value", "domain"}
                else "old-" + name,
                source_digest="3" * 64,
                **common,
            ),
            "replacement": SharedInputEvidence(
                source_receipt=after["value"]
                if name in {"value", "domain"}
                else "new-" + name,
                source_digest="4" * 64,
                **{**common, "source_information_cutoff": date(2024, 1, 9)},
            ),
        }
    args = dict(
        original=original,
        original_inputs=before,
        replacement_inputs=after,
        settings=settings,
        kernel=kernel,
        scope={
            "start": first.isoformat(),
            "end": last.isoformat(),
            "information_cutoff": last.isoformat(),
            "universe": "test",
            "numerical_contract": "5" * 64,
            "read_windows": {
                name: {"start": first.isoformat(), "end": last.isoformat()}
                for name in CHANNELS
            },
        },
        consumed=consumed,
        original_request={"id": "original-request", "digest": "6" * 64},
        report_reference="task://original-and-selected-input-report",
        report_sha256="7" * 64,
        sources={"price": "new"},
    )
    return store, args, frame


def test_primitive_rebinding_preserves_files_and_is_idempotent_without_reads(
    tmp_path, monkeypatch
):
    store, args, _ = setup(tmp_path)

    def forbidden(*a, **kw):
        raise AssertionError(
            "Primitive maintenance read or calculated original numerical values"
        )

    monkeypatch.setattr(store, "read_reference", forbidden)
    monkeypatch.setattr(store, "_save_table", forbidden)
    maintained = store.maintain_shared_inputs("owner", **args)
    assert maintained["files"] == args["original"]["files"]
    assert set(maintained["metadata"]) == {
        "groups",
        "through",
        "primitive_only_input_binding",
    }
    assert maintained["metadata"]["through"] == args["original"]["metadata"]["through"]
    assert store.maintain_shared_inputs("owner", **args) == maintained
    assert (
        store.shared_receipt(
            "owner", "signal", input_identity=args["original"]["input_identity"]
        )
        == args["original"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "value",
        "domain",
        "prices",
        "calendar",
        "labels",
        "schema",
        "selector",
        "missing",
        "partial",
        "none",
        "core_ref",
        "window",
        "through",
        "settings",
        "kernel",
        "report",
    ],
)
def test_maintenance_rejects_changed_or_incomplete_consumption(tmp_path, change):
    store, args, _ = setup(tmp_path)
    if change in CHANNELS:
        args["consumed"][change]["replacement"] = replace(
            args["consumed"][change]["replacement"], content_token="8" * 64
        )
    elif change in {"schema", "selector", "partial", "none", "core_ref", "window"}:
        patch = {
            "schema": {"schema": {"value": "String"}},
            "selector": {
                "selector": {
                    "view": "snapshot",
                    "strict": True,
                    "include_historical_baseline": False,
                }
            },
            "partial": {"complete": False},
            "none": {"content_token": None},
            "core_ref": {"source_receipt": "wrong"},
            "window": {"start": date(2024, 1, 3)},
        }[change]
        args["consumed"]["value"]["replacement"] = replace(
            args["consumed"]["value"]["replacement"], **patch
        )
    elif change == "missing":
        del args["consumed"]["labels"]
    elif change == "through":
        args["replacement_inputs"]["through"] = date(2024, 1, 4)
    elif change == "settings":
        args["settings"]["annualization"] = 252
    elif change == "kernel":
        args["kernel"] = "different.v1"
    elif change == "report":
        args["report_sha256"] = None
    with pytest.raises((ValueError, TypeError)):
        store.maintain_shared_inputs("owner", **args)
    assert store.shared_receipt("owner", "signal") == args["original"]


def test_missing_original_authority_blocks_even_repeated_maintenance(tmp_path):
    store, args, _ = setup(tmp_path)
    new = store.maintain_shared_inputs("owner", **args)
    with store._connect() as connection:
        connection.execute(
            "DELETE FROM bt_shared WHERE receipt_identity=?",
            (args["original"]["receipt_identity"],),
        )
    with pytest.raises(ValueError, match=r"authority|registered"):
        store.maintain_shared_inputs("owner", **args)
    with pytest.raises(ValueError, match="authority"):
        store.shared_receipt("owner", "signal", input_identity=new["input_identity"])


@pytest.mark.parametrize(
    "patch",
    [
        {"schema": {"time": True}},
        {"schema": ["time"]},
        {"schema": {"": "Date"}},
        {"trace_columns": "abc"},
        {"trace_columns": ("available", "available")},
        {"trace_columns": (123,)},
        {"source_receipt": 123},
        {"selector": ["snapshot"]},
    ],
)
def test_both_equal_but_malformed_projections_cannot_be_admitted(tmp_path, patch):
    store, args, _ = setup(tmp_path)
    for side in ("original", "replacement"):
        args["consumed"]["prices"][side] = replace(
            args["consumed"]["prices"][side], **patch
        )
    with pytest.raises(ValueError, match="incomplete typed"):
        store.maintain_shared_inputs("owner", **args)
    assert store.shared_receipt("owner", "signal") == args["original"]


@pytest.mark.parametrize("race", ["authority_removed", "cancellation"])
def test_closing_publication_rechecks_authority_and_cancellation(
    tmp_path, monkeypatch, race
):
    store, args, _ = setup(tmp_path)
    prepare = store._publication_files
    prepared = False

    def prepare_then_change(*values, **kwargs):
        nonlocal prepared
        files = prepare(*values, **kwargs)
        prepared = True
        if race == "authority_removed":
            with store._connect() as connection:
                connection.execute(
                    "DELETE FROM bt_shared WHERE receipt_identity=?",
                    (args["original"]["receipt_identity"],),
                )
        return files

    def cancel():
        if prepared and race == "cancellation":
            raise RuntimeError("canceled at closing publication")

    monkeypatch.setattr(store, "_publication_files", prepare_then_change)
    with pytest.raises((ValueError, RuntimeError), match=r"authority|canceled"):
        store.maintain_shared_inputs("owner", **args, check_canceled=cancel)
    replacement = evaluation_identity(
        "signal",
        inputs=args["replacement_inputs"],
        settings=args["settings"],
        kernel=args["kernel"],
    )
    assert store.shared_receipt("owner", "signal", input_identity=replacement) is None
    if race == "cancellation":
        assert store.shared_receipt("owner", "signal") == args["original"]


def test_generic_publication_cannot_admit_resigned_source_report(tmp_path):
    store, args, _ = setup(tmp_path)
    new = store.maintain_shared_inputs("owner", **args)
    metadata = deepcopy(new["metadata"])
    binding = metadata["primitive_only_input_binding"]
    binding["report_sha256"] = "9" * 64
    binding["identity"] = _identity(
        {k: v for k, v in binding.items() if k != "identity"}
    )
    with pytest.raises(ValueError, match="admission"):
        store.publish_shared(
            "owner",
            "signal",
            new["input_identity"],
            {},
            metadata=metadata,
            sources=new["sources"],
            file_references=new["files"],
            expected_shared=new,
        )
    assert store.shared_receipt("owner", "signal") == new


def test_existing_admission_allows_first_coverage_only_and_remains_repeatable(tmp_path):
    store, args, frame = setup(tmp_path)
    new = store.maintain_shared_inputs("owner", **args)
    coverage = store.publish_shared(
        "owner",
        "signal",
        new["input_identity"],
        {"daily_signal_coverage": frame},
        metadata=new["metadata"],
        sources=new["sources"],
        file_references=new["files"],
        expected_shared=new,
    )
    assert store.maintain_shared_inputs("owner", **args) == coverage
    for patch in (
        {"groups": ["book_tail", "horizons"]},
        {"scope": {"start": "2024-01-02"}},
        {"state": {"through": "2024-01-03"}},
    ):
        with pytest.raises(ValueError, match="Primitive"):
            store.publish_shared(
                "owner",
                "signal",
                coverage["input_identity"],
                {},
                metadata={**coverage["metadata"], **patch},
                sources=coverage["sources"],
                file_references=coverage["files"],
                expected_shared=coverage,
            )


def test_cancellation_and_foreign_latest_preserve_completed_receipts(tmp_path):
    store, args, frame = setup(tmp_path)

    def cancel():
        raise RuntimeError("canceled-test")

    with pytest.raises(RuntimeError, match="canceled-test"):
        store.maintain_shared_inputs("owner", **args, check_canceled=cancel)
    assert store.shared_receipt("owner", "signal") == args["original"]
    foreign = store.publish_shared(
        "owner",
        "signal",
        "foreign",
        {"daily_book_returns": frame},
        metadata={"groups": ["book_tail"], "through": "2024-01-04"},
        expected_shared=args["original"],
    )
    with pytest.raises(RuntimeError, match="changed"):
        store.maintain_shared_inputs("owner", **args)
    assert store.shared_receipt("owner", "signal") == foreign
