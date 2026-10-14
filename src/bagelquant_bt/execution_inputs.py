"""Source-owned complete consumed-input authority for account continuation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import date
from typing import Any

from .exceptions import InputValidationError
from .input_maintenance import SharedInputEvidence

VERSION = "bt.execution.inputs.v1"
CHANNELS = frozenset(
    {"prices", "calendar", "availability", "lot_sizes", "actions", "action_coverage"}
)


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _day(value):
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise InputValidationError("execution input scope requires canonical ISO dates")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise InputValidationError(
            "execution input scope requires ISO dates"
        ) from error
    if parsed.isoformat() != value:
        raise InputValidationError("execution input scope requires canonical ISO dates")
    return parsed


def _channels(mode):
    if mode == "weights":
        return CHANNELS | {"weights", "decisions"}
    if mode == "transactions":
        return CHANNELS | {"transactions"}
    raise InputValidationError("unknown execution input mode")


def _projection(value):
    if isinstance(value, SharedInputEvidence):
        value = asdict(value)
    if not isinstance(value, Mapping) or set(value) != {
        "source_receipt",
        "source_digest",
        "content_token",
        "start",
        "end",
        "source_information_cutoff",
        "schema",
        "selector",
        "complete",
        "trace_columns",
    }:
        raise InputValidationError(
            "execution requires complete typed source projections"
        )
    if (
        not isinstance(value["source_receipt"], str)
        or not value["source_receipt"]
        or not _sha(value["source_digest"])
        or not _sha(value["content_token"])
        or value["complete"] is not True
        or not isinstance(value["schema"], Mapping)
        or not value["schema"]
        or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in value["schema"].items()
        )
        or not isinstance(value["selector"], Mapping)
        or not value["selector"]
        or not isinstance(value["trace_columns"], (tuple, list))
        or len(set(value["trace_columns"])) != len(value["trace_columns"])
        or any(not isinstance(k, str) or not k for k in value["trace_columns"])
    ):
        raise InputValidationError("execution input source projection is incomplete")
    start, end = _day(value["start"]), _day(value["end"])
    information = _day(value["source_information_cutoff"])
    if start > end:
        raise InputValidationError("execution input source scope is reversed")
    selector = value["selector"]
    if "asset_starts" in selector:
        assets, starts = selector.get("assets"), selector["asset_starts"]
        if (
            not isinstance(assets, list)
            or any(not isinstance(asset, str) or not asset for asset in assets)
            or len(set(assets)) != len(assets)
            or not isinstance(starts, Mapping)
            or set(starts) != set(assets)
            or any(not start <= _day(day) <= end for day in starts.values())
        ):
            raise InputValidationError(
                "execution asset admission coordinates are malformed"
            )
    return {
        **dict(value),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "source_information_cutoff": information.isoformat(),
        "schema": dict(value["schema"]),
        "selector": dict(value["selector"]),
        "trace_columns": list(value["trace_columns"]),
    }


def _settings(config):
    return {
        key: value for key, value in config.identity().items() if key != "annualization"
    }


def bind_execution_inputs(
    evidence: Mapping[str, SharedInputEvidence],
    *,
    through: date,
    initial_capital: float,
    config,
    mode: str,
    session_lag: int,
    reference_price_mode: str,
) -> dict[str, Any]:
    """Freeze complete source projections; resource/report options stay neutral.

    The caller obtains each projection from the owning source backend. BT
    validates scope and equality, and never resolves foreign storage. Explicit
    empty channels still have a schema, selector, original source and token.
    """
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != _channels(mode)
        or any(not isinstance(item, SharedInputEvidence) for item in evidence.values())
    ):
        raise InputValidationError("execution input evidence needs every typed channel")
    channels = {name: _projection(item) for name, item in evidence.items()}
    starts = {item["start"] for item in channels.values()}
    if len(starts) != 1 or any(
        _day(item["start"]) > through or _day(item["end"]) < through
        for item in channels.values()
    ):
        raise InputValidationError(
            "execution source evidence does not cover complete history"
        )
    binding = {
        "version": VERSION,
        "mode": mode,
        "through": through.isoformat(),
        "initial_capital": initial_capital,
        "settings": _settings(config),
        "session_lag": session_lag,
        "reference_price_mode": reference_price_mode,
        "channels": channels,
    }
    try:
        return json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise InputValidationError(
            "execution source selectors must be finite JSON"
        ) from error


def validate_execution_continuation(
    checkpoint,
    *,
    initial_capital: float,
    config,
    prefix_evidence: Mapping[str, SharedInputEvidence],
    mode: str = "weights",
    session_lag: int = 1,
    reference_price_mode: str = "execution",
) -> dict[str, Any]:
    """Validate recorded complete prefixes before any numerical input read.

    ``checkpoint`` may be the public checkpoint or its ``save_checkpoint``
    metadata. This operation only compares recorded authority. Receipt IDs may
    change; consumed coordinates, payload, nulls, traces and scope must not.
    Legacy row-hash checkpoints need explicit original-input maintenance.
    """
    binding = (
        checkpoint.get("source_binding")
        if isinstance(checkpoint, Mapping)
        else checkpoint.source_binding
    )
    boundary = (
        checkpoint.get("time") if isinstance(checkpoint, Mapping) else checkpoint.time
    )
    if not isinstance(binding, Mapping) or binding.get("version") != VERSION:
        raise InputValidationError(
            "execution checkpoint lacks complete source authority; "
            "explicit original-input checkpoint maintenance is required"
        )
    if (
        set(binding)
        != {
            "version",
            "mode",
            "through",
            "initial_capital",
            "settings",
            "session_lag",
            "reference_price_mode",
            "channels",
        }
        or _day(binding["through"]) != _day(boundary)
        or binding["mode"] != mode
        or binding["initial_capital"] != initial_capital
        or binding["settings"] != _settings(config)
        or binding["session_lag"] != session_lag
        or binding["reference_price_mode"] != reference_price_mode
    ):
        raise InputValidationError(
            "execution checkpoint scope or numerical settings differ"
        )
    recorded_hash = (
        checkpoint.get("prefix_hash")
        if isinstance(checkpoint, Mapping)
        else checkpoint.prefix_hash
    )
    try:
        actual_hash = hashlib.sha256(
            json.dumps(
                binding, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    except (ValueError, TypeError) as error:
        raise InputValidationError(
            "execution checkpoint source authority is malformed"
        ) from error
    if recorded_hash != actual_hash:
        raise InputValidationError(
            "execution checkpoint source authority digest differs"
        )
    originals = binding["channels"]
    if (
        not isinstance(originals, Mapping)
        or set(originals) != _channels(mode)
        or not isinstance(prefix_evidence, Mapping)
        or set(prefix_evidence) != set(originals)
        or any(
            not isinstance(item, SharedInputEvidence)
            for item in prefix_evidence.values()
        )
    ):
        raise InputValidationError(
            "execution prefix evidence needs every typed channel"
        )
    semantic = ("content_token", "start", "end", "schema", "selector", "trace_columns")
    for name, original in originals.items():
        before, after = _projection(original), _projection(prefix_evidence[name])
        if _day(before["end"]) < _day(boundary) or any(
            before[key] != after[key] for key in semantic
        ):
            raise InputValidationError(
                f"execution checkpoint consumed {name} prefix changed"
            )
    return dict(binding)
