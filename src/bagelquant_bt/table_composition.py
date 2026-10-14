"""Immutable selections and concatenations of BT-owned original tables."""

from __future__ import annotations

import polars as pl

COORDINATES = (
    "time",
    "interval_start",
    "interval_end",
    "checkpoint_date",
    "available_date",
    "asset_id",
    "ledger",
    "execution_date",
    "decision_date",
    "exit_date",
    "entry_date",
    "record_date",
    "evaluation_date",
    "window_kind",
    "window_id",
    "lag",
    "path_kind",
    "method",
    "quantile",
    "horizon_sessions",
    "target_date",
)


def coordinates(frame: pl.DataFrame) -> list[dict] | None:
    """Record exact selection counts while newly published rows are consumed."""
    columns = [name for name in COORDINATES if name in frame.columns]
    if not columns:
        return None
    grouped = frame.group_by(columns).len(name="_rows").sort(columns)
    return [
        {
            name: value.isoformat() if hasattr(value, "isoformat") else value
            for name, value in row.items()
        }
        for row in grouped.to_dicts()
    ]


def filters(values, available):
    result = []
    for value in values:
        if (
            set(value) != {"column", "operation", "value"}
            or value["column"] not in available
            or value["operation"] not in {"eq", "ge", "le"}
        ):
            raise ValueError("BT table selection needs indexed coordinates")
        item = value["value"]
        if hasattr(item, "isoformat"):
            item = item.isoformat()
        if not isinstance(item, (str, int)) or isinstance(item, bool):
            raise ValueError("BT coordinate selections require Date/String/Integer")
        if isinstance(available, dict):
            dtype = available[value["column"]]
            if dtype == "Date":
                from datetime import date

                if not isinstance(item, str):
                    raise ValueError("BT Date coordinate requires an ISO date")
                try:
                    item = date.fromisoformat(item).isoformat()
                except ValueError as error:
                    raise ValueError(
                        "BT Date coordinate requires an ISO date"
                    ) from error
            elif dtype == "String" and not isinstance(item, str):
                raise ValueError("BT String coordinate requires a string")
            elif dtype.startswith(("Int", "UInt")) and not isinstance(item, int):
                raise ValueError("BT Integer coordinate requires an integer")
        result.append({**value, "value": item})
    return result


def select(frame, predicates, *, serialized=False):
    for item in predicates:
        column = pl.col(item["column"])
        value = item["value"]
        schema = (
            frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
        )
        if not serialized and schema[item["column"]] == pl.Date:
            from datetime import date

            value = date.fromisoformat(value)
        if item["operation"] == "eq":
            condition = column == value
        elif item["operation"] == "ge":
            condition = column >= value
        else:
            condition = column <= value
        frame = frame.filter(condition)
    return frame


def _coordinate_frame(facts, schema=None):
    overrides = {"_rows": pl.UInt64}
    for name, dtype in (schema or {}).items():
        if name in COORDINATES and name in facts[0]:
            if dtype in {"Date", "String"}:
                overrides[name] = pl.String
            elif dtype.startswith(("Int", "UInt")):
                overrides[name] = getattr(pl, dtype)
    return pl.DataFrame(facts, schema_overrides=overrides, infer_schema_length=None)


def selected_coordinates(facts, predicates, *, schema=None):
    if not predicates:
        return facts
    if facts is None:
        raise ValueError(
            "BT selected references need explicit coordinate-index maintenance"
        )
    if not facts:
        return []
    if any(item["column"] not in facts[0] for item in predicates):
        raise ValueError(
            "BT selected references need explicit coordinate-index maintenance"
        )
    return select(
        _coordinate_frame(facts, schema), predicates, serialized=True
    ).to_dicts()


def admit_coordinates(keys, schema):
    """Validate exact typed keys against the original registered coordinate axes."""
    if not isinstance(keys, pl.DataFrame) or not keys.columns:
        raise ValueError("BT coordinate keys require a typed nonempty column set")
    for name, dtype in keys.schema.items():
        if (
            name not in COORDINATES
            or schema.get(name) != str(dtype)
            or not (dtype in (pl.Date, pl.String) or dtype.is_integer())
            or keys[name].null_count()
        ):
            raise ValueError(
                "BT coordinate keys require original indexed nonnull types"
            )
    return keys.unique(maintain_order=True)


def select_keys(frame, keys, *, serialized=False):
    """Push each key axis into Parquet, then retain only the exact key tuples."""
    columns = keys.columns
    if serialized:
        keys = keys.with_columns(
            pl.col(name).dt.to_string("%Y-%m-%d")
            for name, dtype in keys.schema.items()
            if dtype == pl.Date
        )
    for name in columns:
        frame = frame.filter(pl.col(name).is_in(keys[name].implode()))
    return frame.join(
        keys.lazy() if isinstance(frame, pl.LazyFrame) else keys,
        on=columns,
        how="semi",
    )


def selected_keys(facts, keys, *, schema=None):
    if facts is None or (facts and not set(keys.columns).issubset(facts[0])):
        raise ValueError(
            "BT selected references need explicit coordinate-index maintenance"
        )
    if not facts or keys.is_empty():
        return []
    return select_keys(
        _coordinate_frame(facts, schema), keys, serialized=True
    ).to_dicts()


def combine_coordinates(facts, *, schema=None):
    if not facts:
        return []
    columns = [name for name in COORDINATES if name in facts[0]]
    return (
        _coordinate_frame(facts, schema)
        .group_by(columns)
        .agg(pl.col("_rows").sum())
        .sort(columns)
        .to_dicts()
    )
