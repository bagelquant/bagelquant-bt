"""Deterministic whole-lot allocation for one target-weight snapshot."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, milp

from .exceptions import InputValidationError

_BUDGET_TOLERANCE = 1e-7


@dataclass(frozen=True, slots=True)
class IntegerTargetAllocation:
    """Whole-lot positions and budget summary for one target snapshot."""

    positions: pl.DataFrame
    stock_exposure: float
    stock_budget: float
    allocated_notional: float
    residual_cash: float


def allocate_integer_positions(
    target_weights: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    total_notional: float,
    lot_sizes: pl.DataFrame | None = None,
    minimum_quantities: pl.DataFrame | None = None,
    allow_one_lot_over_target: bool = True,
) -> IntegerTargetAllocation:
    """Allocate a target snapshot into deterministic integer-lot positions.

    The solver first maximizes deployed stock notional without exceeding the
    target stock budget. Among equally deployed solutions it minimizes the
    absolute notional deviation from the continuous target. Existing minimum
    quantities may be arbitrary integers; any incremental quantity uses the
    declared lot size.
    """

    if not math.isfinite(total_notional) or total_notional <= 0:
        raise InputValidationError("total_notional must be positive and finite")
    targets = _targets(target_weights)
    price_by_asset = _positive_values(prices, "price", label="prices")
    lot_by_asset = _positive_integers(
        lot_sizes,
        "lot_size",
        label="lot_sizes",
        default=1,
    )
    minimum_by_asset = _nonnegative_integers(
        minimum_quantities,
        "minimum_quantity",
        label="minimum_quantities",
    )
    assets = sorted(set(targets) | set(minimum_by_asset))
    if not assets:
        return IntegerTargetAllocation(
            positions=_empty_positions(),
            stock_exposure=0.0,
            stock_budget=0.0,
            allocated_notional=0.0,
            residual_cash=0.0,
        )
    missing_prices = [asset_id for asset_id in assets if asset_id not in price_by_asset]
    if missing_prices:
        raise InputValidationError(
            "allocation requires a finite positive price for: "
            + ", ".join(missing_prices)
        )

    stock_exposure = float(sum(targets.values()))
    if stock_exposure > 1.0 + _BUDGET_TOLERANCE:
        raise InputValidationError("target weights must sum to at most one")
    stock_budget = total_notional * min(stock_exposure, 1.0)
    minimum_value = sum(
        minimum_by_asset.get(asset_id, 0) * price_by_asset[asset_id]
        for asset_id in assets
    )
    if minimum_value > stock_budget + _BUDGET_TOLERANCE:
        raise InputValidationError("minimum positions exceed the target stock budget")

    minimums: list[int] = []
    lots: list[int] = []
    prices_array: list[float] = []
    ideal_values: list[float] = []
    maximum_lot_counts: list[int] = []
    for asset_id in assets:
        minimum = minimum_by_asset.get(asset_id, 0)
        lot = lot_by_asset.get(asset_id, 1)
        price = price_by_asset[asset_id]
        ideal_value = targets.get(asset_id, 0.0) * total_notional
        remaining_ideal_quantity = max(ideal_value / price - minimum, 0.0)
        if allow_one_lot_over_target:
            maximum_lots = math.ceil(max(remaining_ideal_quantity / lot - 1e-12, 0.0))
        else:
            maximum_lots = math.floor(max(remaining_ideal_quantity / lot + 1e-12, 0.0))
        minimums.append(minimum)
        lots.append(lot)
        prices_array.append(price)
        ideal_values.append(ideal_value)
        maximum_lot_counts.append(maximum_lots)

    lot_values = np.asarray(lots, dtype=float) * np.asarray(prices_array, dtype=float)
    remaining_budget = max(stock_budget - minimum_value, 0.0)
    lot_counts = _solve_lot_counts(
        assets=assets,
        lot_values=lot_values,
        maximum_lot_counts=np.asarray(maximum_lot_counts, dtype=float),
        minimum_values=np.asarray(minimums, dtype=float)
        * np.asarray(prices_array, dtype=float),
        ideal_values=np.asarray(ideal_values, dtype=float),
        remaining_budget=remaining_budget,
    )

    rows: list[dict[str, object]] = []
    allocated_notional = 0.0
    for index, asset_id in enumerate(assets):
        quantity = minimums[index] + int(lot_counts[index]) * lots[index]
        actual_notional = quantity * prices_array[index]
        allocated_notional += actual_notional
        ideal_notional = ideal_values[index]
        rows.append(
            {
                "asset_id": asset_id,
                "target_weight": targets.get(asset_id, 0.0),
                "price": prices_array[index],
                "lot_size": lots[index],
                "minimum_quantity": minimums[index],
                "target_quantity": quantity,
                "target_notional": ideal_notional,
                "allocated_notional": actual_notional,
                "allocated_weight": actual_notional / total_notional,
                "notional_deviation": actual_notional - ideal_notional,
            }
        )
    residual_cash = stock_budget - allocated_notional
    if residual_cash < -_BUDGET_TOLERANCE:
        raise RuntimeError("integer allocation exceeded its stock budget")
    return IntegerTargetAllocation(
        positions=pl.DataFrame(rows, schema=_position_schema()).sort("asset_id"),
        stock_exposure=stock_exposure,
        stock_budget=stock_budget,
        allocated_notional=allocated_notional,
        residual_cash=max(residual_cash, 0.0),
    )


def _solve_lot_counts(
    *,
    assets: list[str],
    lot_values: np.ndarray,
    maximum_lot_counts: np.ndarray,
    minimum_values: np.ndarray,
    ideal_values: np.ndarray,
    remaining_budget: float,
) -> np.ndarray:
    count = len(assets)
    bounds = Bounds(np.zeros(count), maximum_lot_counts)
    budget = LinearConstraint(
        lot_values.reshape(1, -1),
        -np.inf,
        remaining_budget,
    )
    first = milp(
        -lot_values,
        integrality=np.ones(count),
        bounds=bounds,
        constraints=budget,
        options={"disp": False},
    )
    if not first.success or first.x is None:
        raise InputValidationError(
            "whole-lot allocation could not maximize the stock budget"
            + _solver_failure_detail(first)
        )
    maximum_deployment = float(lot_values @ np.rint(first.x))

    variable_count = count * 2
    objective = np.concatenate(
        (
            np.asarray([(index + 1) * 1e-10 for index in range(count)], dtype=float),
            np.ones(count),
        )
    )
    integrality = np.concatenate((np.ones(count), np.zeros(count)))
    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.concatenate((maximum_lot_counts, np.full(count, np.inf)))
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    budget_row = np.zeros(variable_count)
    budget_row[:count] = lot_values
    rows.append(budget_row)
    lower.append(maximum_deployment - _BUDGET_TOLERANCE)
    upper.append(remaining_budget)

    target_gap = ideal_values - minimum_values
    for index in range(count):
        positive = np.zeros(variable_count)
        positive[index] = lot_values[index]
        positive[count + index] = -1.0
        rows.append(positive)
        lower.append(-np.inf)
        upper.append(target_gap[index])

        negative = np.zeros(variable_count)
        negative[index] = -lot_values[index]
        negative[count + index] = -1.0
        rows.append(negative)
        lower.append(-np.inf)
        upper.append(-target_gap[index])

    second = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(np.vstack(rows), lower, upper),
        options={"disp": False},
    )
    if not second.success or second.x is None:
        raise InputValidationError(
            "whole-lot allocation could not minimize target deviation"
            + _solver_failure_detail(second)
        )
    result = np.rint(second.x[:count]).astype(np.int64)
    if np.any(result < 0) or np.any(result > maximum_lot_counts + 1e-8):
        raise RuntimeError("whole-lot allocation returned invalid lot counts")
    return result


def _solver_failure_detail(result: object) -> str:
    """Preserve HiGHS status without exposing solver internals elsewhere."""

    status = getattr(result, "status", None)
    message = str(getattr(result, "message", "") or "").strip()
    values = []
    if status is not None:
        values.append(f"status={status}")
    if message:
        values.append(f"message={message}")
    return "" if not values else f" ({', '.join(values)})"


def _targets(frame: pl.DataFrame) -> dict[str, float]:
    if not {"asset_id", "weight"}.issubset(frame.columns):
        raise InputValidationError("target_weights requires asset_id and weight")
    if frame.n_unique(subset=["asset_id"]) != frame.height:
        raise InputValidationError("target_weights asset_id values must be unique")
    result: dict[str, float] = {}
    for row in frame.select("asset_id", "weight").iter_rows(named=True):
        asset_id = str(row["asset_id"])
        value = float(row["weight"])
        if not math.isfinite(value) or value < 0:
            raise InputValidationError("target weights must be finite and nonnegative")
        if value > 0:
            result[asset_id] = value
    return result


def _positive_values(
    frame: pl.DataFrame,
    column: str,
    *,
    label: str,
) -> dict[str, float]:
    if not {"asset_id", column}.issubset(frame.columns):
        raise InputValidationError(f"{label} requires asset_id and {column}")
    if frame.n_unique(subset=["asset_id"]) != frame.height:
        raise InputValidationError(f"{label} asset_id values must be unique")
    result: dict[str, float] = {}
    for row in frame.select("asset_id", column).iter_rows(named=True):
        value = float(row[column])
        if not math.isfinite(value) or value <= 0:
            raise InputValidationError(f"{column} must be finite and positive")
        result[str(row["asset_id"])] = value
    return result


def _positive_integers(
    frame: pl.DataFrame | None,
    column: str,
    *,
    label: str,
    default: int,
) -> dict[str, int]:
    if frame is None:
        return {}
    result = _integer_values(frame, column, label=label, minimum=1)
    return {asset_id: value or default for asset_id, value in result.items()}


def _nonnegative_integers(
    frame: pl.DataFrame | None,
    column: str,
    *,
    label: str,
) -> dict[str, int]:
    if frame is None:
        return {}
    return _integer_values(frame, column, label=label, minimum=0)


def _integer_values(
    frame: pl.DataFrame,
    column: str,
    *,
    label: str,
    minimum: int,
) -> dict[str, int]:
    if not {"asset_id", column}.issubset(frame.columns):
        raise InputValidationError(f"{label} requires asset_id and {column}")
    if frame.n_unique(subset=["asset_id"]) != frame.height:
        raise InputValidationError(f"{label} asset_id values must be unique")
    result: dict[str, int] = {}
    for row in frame.select("asset_id", column).iter_rows(named=True):
        raw = row[column]
        value = int(raw)
        if isinstance(raw, bool) or value != raw or value < minimum:
            raise InputValidationError(
                f"{column} must be an integer greater than or equal to {minimum}"
            )
        result[str(row["asset_id"])] = value
    return result


def _position_schema() -> dict[str, pl.DataType]:
    return {
        "asset_id": pl.String,
        "target_weight": pl.Float64,
        "price": pl.Float64,
        "lot_size": pl.Int64,
        "minimum_quantity": pl.Int64,
        "target_quantity": pl.Int64,
        "target_notional": pl.Float64,
        "allocated_notional": pl.Float64,
        "allocated_weight": pl.Float64,
        "notional_deviation": pl.Float64,
    }


def _empty_positions() -> pl.DataFrame:
    return pl.DataFrame(schema=_position_schema())


__all__ = ["IntegerTargetAllocation", "allocate_integer_positions"]
