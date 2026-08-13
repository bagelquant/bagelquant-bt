"""Provider-neutral valuation for an observed live account sleeve."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from .exceptions import InputValidationError
from .inputs import ASSET_ID


@dataclass(frozen=True, slots=True)
class ObservedAccountValuation:
    """One immutable valuation point and the marks needed by the next point."""

    observed_at: datetime
    cash: float
    receivables: float
    position_value: float
    equity: float
    external_flow: float
    units: float
    nav: float
    flow_neutral_return: float
    stale_assets: tuple[str, ...]
    marks: pl.DataFrame


def value_observed_account(
    positions: pl.DataFrame,
    *,
    observed_at: datetime,
    cash: float,
    receivables: float = 0.0,
    external_flow: float = 0.0,
    previous: ObservedAccountValuation | None = None,
    nav_base: float = 1.0,
) -> ObservedAccountValuation:
    """Value observed quantities with flow-neutral fund units.

    ``positions`` contains unique ``asset_id``, ``quantity`` and nullable
    unadjusted ``price`` columns. A missing price carries the previous observed
    mark and marks the asset stale. A new positive position without any mark
    fails rather than receiving a fabricated value.
    """

    for name, value in (
        ("cash", cash),
        ("receivables", receivables),
        ("external_flow", external_flow),
        ("nav_base", nav_base),
    ):
        if not math.isfinite(value):
            raise InputValidationError(f"{name} must be finite")
    if cash < 0 or receivables < 0 or nav_base <= 0:
        raise InputValidationError(
            "cash and receivables must be nonnegative and nav_base must be positive"
        )
    if previous is not None and observed_at <= previous.observed_at:
        raise InputValidationError("observed_at must advance after the previous point")

    canonical = _canonical_positions(positions)
    prior_marks = (
        {}
        if previous is None
        else {
            str(row[ASSET_ID]): float(row["price"])
            for row in previous.marks.iter_rows(named=True)
        }
    )
    marked_rows: list[dict[str, object]] = []
    stale_assets: list[str] = []
    for row in canonical.iter_rows(named=True):
        asset_id = str(row[ASSET_ID])
        quantity = float(row["quantity"])
        observed_price = row["price"]
        stale = observed_price is None
        if observed_price is None:
            observed_price = prior_marks.get(asset_id)
        if observed_price is None and quantity > 0:
            raise InputValidationError(
                f"positive position {asset_id!r} has no current or previous price"
            )
        price = 0.0 if observed_price is None else float(observed_price)
        if stale and quantity > 0:
            stale_assets.append(asset_id)
        marked_rows.append(
            {
                ASSET_ID: asset_id,
                "quantity": quantity,
                "price": price,
                "market_value": quantity * price,
                "is_stale": stale and quantity > 0,
            }
        )

    marks = pl.DataFrame(
        marked_rows,
        schema={
            ASSET_ID: pl.String,
            "quantity": pl.Float64,
            "price": pl.Float64,
            "market_value": pl.Float64,
            "is_stale": pl.Boolean,
        },
    ).sort(ASSET_ID)
    position_value = float(marks.get_column("market_value").sum())
    equity = float(cash + receivables + position_value)
    if equity <= 0:
        raise InputValidationError("observed account equity must be positive")

    if previous is None:
        if abs(external_flow) > 1e-12:
            raise InputValidationError(
                "the initial valuation cannot contain external_flow"
            )
        units = equity / nav_base
        nav = nav_base
        flow_neutral_return = 0.0
    else:
        units = previous.units + external_flow / previous.nav
        if units <= 0:
            raise InputValidationError(
                "external_flow would redeem all or more fund units"
            )
        nav = equity / units
        flow_neutral_return = nav / previous.nav - 1.0

    return ObservedAccountValuation(
        observed_at=observed_at,
        cash=float(cash),
        receivables=float(receivables),
        position_value=position_value,
        equity=equity,
        external_flow=float(external_flow),
        units=float(units),
        nav=float(nav),
        flow_neutral_return=float(flow_neutral_return),
        stale_assets=tuple(sorted(stale_assets)),
        marks=marks,
    )


def _canonical_positions(frame: pl.DataFrame) -> pl.DataFrame:
    required = {ASSET_ID, "quantity", "price"}
    if not isinstance(frame, pl.DataFrame) or not required.issubset(frame.columns):
        raise InputValidationError(
            "positions requires asset_id, quantity, and nullable price columns"
        )
    result = frame.select(ASSET_ID, "quantity", "price").with_columns(
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("quantity").cast(pl.Float64, strict=False),
        pl.col("price").cast(pl.Float64, strict=False),
    )
    if result.select(pl.col(ASSET_ID).is_duplicated().any()).item():
        raise InputValidationError("positions must be unique by asset_id")
    invalid = result.filter(
        pl.col(ASSET_ID).str.strip_chars().eq("")
        | pl.col("quantity").is_null()
        | ~pl.col("quantity").is_finite()
        | (pl.col("quantity") < 0)
        | (
            pl.col("price").is_not_null()
            & (~pl.col("price").is_finite() | (pl.col("price") <= 0))
        )
    )
    if invalid.height:
        raise InputValidationError(
            "position quantities must be finite and nonnegative; prices must be "
            "null or finite and positive"
        )
    return result.sort(ASSET_ID)


__all__ = ["ObservedAccountValuation", "value_observed_account"]
