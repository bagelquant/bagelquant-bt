"""Authoritative same-boundary admission for retained legacy signal primitives.

Explicit original recovery and ordinary proven unchanged reuse share this
single metadata admission. Neither path fabricates native continuation state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True, slots=True)
class SharedInputEvidence:
    """A caller/source-owner attestation of a complete typed consumed channel.

    BT checks scope and equality; it cannot independently resolve another
    backend's source semantics. A projection covers missing coordinates,
    null/nonfinite distinctions and the declared traces, not output statistics.
    Source information boundaries remain separate from evaluator maturity.
    """

    source_receipt: str
    source_digest: str
    content_token: str
    start: date
    end: date
    source_information_cutoff: date
    schema: Mapping[str, str]
    selector: Mapping[str, Any]
    complete: bool
    trace_columns: tuple[str, ...] = ()


POLICY = "bt.primitive_only_input_binding.v1"
CHANNELS = frozenset({"value", "domain", "prices", "calendar", "labels"})


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate(original, binding):
    try:
        return _validate_body(original, binding)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(
            "Primitive maintenance declaration is incomplete or malformed"
        ) from error


def _validate_body(original, binding):
    from .store import _identity, evaluation_identity

    if (
        binding.get("policy") != POLICY
        or not _sha(binding.get("report_sha256"))
        or not isinstance(binding.get("report_reference"), str)
        or not binding["report_reference"]
    ):
        raise ValueError(
            "Primitive maintenance requires a source-owned report reference and SHA256"
        )
    if binding.get("original_receipt_identity") != original["receipt_identity"]:
        raise ValueError("Primitive maintenance original receipt differs")
    request = binding["original_request"]
    if not request.get("id") or not _sha(request.get("digest")):
        raise ValueError(
            "Primitive maintenance requires the original immutable execution request"
        )
    before, after = binding["original_inputs"], binding["replacement_inputs"]
    if set(before) != {"node", "value", "sources", "through"} or set(after) != set(
        before
    ):
        raise ValueError(
            "Primitive maintenance needs original/replacement signal identities"
        )
    if (
        not before["node"]
        or before["node"] != after["node"]
        or before["through"] != after["through"]
        or before["through"] != original["metadata"]["through"]
        or not _sha(before["value"])
        or not _sha(after["value"])
    ):
        raise ValueError(
            "Primitive maintenance cannot change node or completed observation boundary"
        )
    if original["metadata"].get("state") or original["metadata"].get("scope"):
        raise ValueError(
            "Primitive-only maintenance cannot rebind native causal continuation state"
        )
    settings, kernel = binding["settings"], binding["kernel"]
    if (
        evaluation_identity("signal", inputs=before, settings=settings, kernel=kernel)
        != original["input_identity"]
    ):
        raise ValueError(
            "Primitive maintenance does not reproduce the original numerical identity"
        )
    scope = binding["scope"]
    if (
        set(scope)
        != {
            "start",
            "end",
            "information_cutoff",
            "universe",
            "numerical_contract",
            "read_windows",
        }
        or scope["end"] != before["through"]
        or scope["information_cutoff"] != before["through"]
        or not scope["universe"]
        or not _sha(scope["numerical_contract"])
    ):
        raise ValueError(
            "Primitive maintenance needs the original evaluator and maturity boundary"
        )
    if date.fromisoformat(scope["start"]) > date.fromisoformat(scope["end"]):
        raise ValueError("Primitive maintenance observation bounds are reversed")
    evidence = binding["consumed"]
    if set(evidence) != CHANNELS or set(scope["read_windows"]) != CHANNELS:
        raise ValueError(
            "Primitive maintenance needs value/domain/price/calendar/label channels"
        )
    for name, pair in evidence.items():
        if set(pair) != {"original", "replacement"}:
            raise ValueError(
                "Primitive maintenance requires two complete source projections"
            )
        for side in pair.values():
            if (
                not isinstance(side["source_receipt"], str)
                or not side["source_receipt"]
                or not _sha(side["source_digest"])
                or not _sha(side["content_token"])
                or not isinstance(side["schema"], Mapping)
                or not side["schema"]
                or any(
                    not isinstance(k, str) or not k or not isinstance(v, str) or not v
                    for k, v in side["schema"].items()
                )
                or not isinstance(side["selector"], Mapping)
                or not side["selector"]
                or side["complete"] is not True
                or not isinstance(side["selector"].get("view"), str)
                or not side["selector"]["view"]
                or type(side["selector"].get("strict")) is not bool
                or type(side["selector"].get("include_historical_baseline")) is not bool
                or not isinstance(side["trace_columns"], (list, tuple))
                or any(
                    not isinstance(name, str) or not name
                    for name in side["trace_columns"]
                )
                or len(set(side["trace_columns"])) != len(side["trace_columns"])
            ):
                raise ValueError(
                    "Primitive maintenance contains incomplete typed source evidence"
                )
            if (
                date.fromisoformat(side["end"]) < date.fromisoformat(side["start"])
                or {k: side[k] for k in ("start", "end")} != scope["read_windows"][name]
            ):
                raise ValueError(
                    "Primitive projection does not cover the admitted read window"
                )
            date.fromisoformat(side["source_information_cutoff"])
        semantic = (
            "content_token",
            "start",
            "end",
            "schema",
            "selector",
            "trace_columns",
        )
        if any(pair["original"][key] != pair["replacement"][key] for key in semantic):
            raise ValueError(f"Primitive maintenance consumed {name} changed")
    for name in ("value", "domain"):
        if (
            evidence[name]["original"]["source_receipt"] != before["value"]
            or evidence[name]["replacement"]["source_receipt"] != after["value"]
        ):
            raise ValueError(
                "Primitive maintenance Core source does not bind its materialization"
            )
    body = {key: value for key, value in binding.items() if key != "identity"}
    if binding.get("identity") != _identity(body):
        raise ValueError("Primitive maintenance admission identity differs")


def verify_binding(connection, receipt, *, visited=()):
    """Check retained immutable authority even before any existing-cache return."""
    from .store import _decode_record

    binding = receipt.get("metadata", {}).get("primitive_only_input_binding")
    if binding is None:
        return
    if not isinstance(binding, Mapping) or not binding.get("original_receipt_identity"):
        raise ValueError("Primitive maintenance authority declaration is malformed")
    identity = receipt["receipt_identity"]
    if identity in visited or binding["original_receipt_identity"] == identity:
        raise ValueError("Primitive maintenance authority contains a cycle")
    row = connection.execute(
        "SELECT * FROM bt_shared WHERE owner_id=? AND component=? "
        "AND receipt_identity=?",
        (
            receipt["owner_id"],
            receipt["component"],
            binding["original_receipt_identity"],
        ),
    ).fetchone()
    if row is None:
        raise ValueError("Primitive maintenance original authority is missing")
    original = _decode_record(
        row,
        "bt.shared.v1",
        owner_id=receipt["owner_id"],
        component=receipt["component"],
        input_identity=row["input_identity"],
        receipt_identity=row["receipt_identity"],
    )
    verify_binding(connection, original, visited=(*visited, identity))
    _validate(original, binding)
    from .store import evaluation_identity

    if (
        receipt["component"] != "signal"
        or receipt["input_identity"]
        != evaluation_identity(
            "signal",
            inputs=binding["replacement_inputs"],
            settings=binding["settings"],
            kernel=binding["kernel"],
        )
        or receipt["sources"] != binding["replacement_sources"]
        or {
            k: v
            for k, v in receipt["metadata"].items()
            if k != "primitive_only_input_binding"
        }
        != {
            k: v
            for k, v in original["metadata"].items()
            if k != "primitive_only_input_binding"
        }
        or any(
            receipt["files"].get(name) != ref for name, ref in original["files"].items()
        )
        or set(receipt["files"]) - set(original["files"]) - {"daily_signal_coverage"}
    ):
        raise ValueError(
            "Primitive admission cannot change old files/groups/scope/state/boundary"
        )


def maintain(
    store,
    owner_id,
    *,
    original,
    original_inputs,
    replacement_inputs,
    settings,
    kernel,
    scope,
    consumed,
    original_request,
    report_reference,
    report_sha256,
    sources,
    check_canceled,
):
    import json

    from .store import _identity, _json, evaluation_identity

    check_canceled()
    if (
        not isinstance(consumed, Mapping)
        or any(
            not isinstance(pair, Mapping) or set(pair) != {"original", "replacement"}
            for pair in consumed.values()
        )
        or any(
            not isinstance(value, SharedInputEvidence)
            for pair in consumed.values()
            for value in pair.values()
        )
    ):
        raise ValueError("Primitive maintenance requires typed SharedInputEvidence")
    from .store import _decode_record

    with store._read_connection() as connection:
        row = connection.execute(
            "SELECT * FROM bt_shared WHERE owner_id=? AND component='signal' "
            "AND receipt_identity=?",
            (owner_id, original["receipt_identity"]),
        ).fetchone()
        actual = (
            None
            if row is None
            else _decode_record(
                row,
                "bt.shared.v1",
                owner_id=owner_id,
                component="signal",
                input_identity=row["input_identity"],
                receipt_identity=row["receipt_identity"],
            )
        )
        if actual is not None:
            verify_binding(connection, actual)
    if actual != original:
        raise ValueError("Primitive maintenance original receipt is not registered")
    binding = {
        "policy": POLICY,
        "original_receipt_identity": original["receipt_identity"],
        "original_inputs": original_inputs,
        "replacement_inputs": replacement_inputs,
        "settings": settings,
        "kernel": kernel,
        "scope": scope,
        "original_request": original_request,
        "report_reference": report_reference,
        "report_sha256": report_sha256,
        "replacement_sources": sources,
        "consumed": {
            name: {side: asdict(value) for side, value in pair.items()}
            for name, pair in consumed.items()
        },
    }
    # Freeze the admitted declaration once; caller mutation cannot change the
    # closing publication. The immutable shared receipt is its admission owner.
    binding = json.loads(_json(binding))
    binding["identity"] = _identity(binding)
    _validate(original, binding)
    store._require_publication_files(original["files"])
    for reference in original["files"].values():
        store.describe_reference(reference)
    replacement = evaluation_identity(
        "signal", inputs=replacement_inputs, settings=settings, kernel=kernel
    )
    if replacement == original["input_identity"]:
        from .store import _decode_record

        with store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            check_canceled()
            row = connection.execute(
                "SELECT * FROM bt_shared WHERE owner_id=? AND component='signal' "
                "AND receipt_identity=?",
                (owner_id, original["receipt_identity"]),
            ).fetchone()
            retained = (
                None
                if row is None
                else _decode_record(
                    row,
                    "bt.shared.v1",
                    owner_id=owner_id,
                    component="signal",
                    input_identity=row["input_identity"],
                    receipt_identity=row["receipt_identity"],
                )
            )
            if retained != original:
                raise ValueError("Primitive maintenance original authority changed")
            verify_binding(connection, retained)
            store._require_publication_files(retained["files"])
            return retained
    expected = store.shared_receipt(owner_id, "signal")
    files = original["files"]
    if expected != original:
        admitted = (
            (expected or {}).get("metadata", {}).get("primitive_only_input_binding")
        )
        if admitted is None or admitted != binding:
            raise RuntimeError("BT shared primitives changed before input maintenance")
        files = expected["files"]
    return store.publish_shared(
        owner_id,
        "signal",
        replacement,
        {},
        metadata={**original["metadata"], "primitive_only_input_binding": binding},
        sources=sources,
        expected_shared=expected,
        file_references=files,
        check_canceled=check_canceled,
        _maintenance_admission=binding,
    )
