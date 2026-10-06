"""Private account state, fills, settlement and corporate-action mechanics.

Signed-share scheduling and the sole execution loop live in
execution_evaluation.py. These primitives never choose market timing or policy.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Literal

import polars as pl

from .allocation import allocate_integer_positions
from .config import TransactionCostConfig
from .exceptions import BacktestConfigError, InputValidationError
from .inputs import ASSET_ID, TIME


@dataclass(frozen=True, slots=True)
class AccountBacktestConfig:
    """Capital, settlement, lot, and cost rules for a whole-share account."""

    capital_mode: Literal["fixed_notional", "compounding"] = "fixed_notional"
    initial_capital: float = 500_000.0
    fixed_notional: float = 500_000.0
    nav_base: float = 1.0
    default_buy_lot_size: int = 1
    settlement_sessions: int = 0
    retry_blocked_orders: bool = True
    transaction_cost: TransactionCostConfig = field(
        default_factory=TransactionCostConfig
    )

    def __post_init__(self) -> None:
        if self.capital_mode not in {"fixed_notional", "compounding"}:
            raise BacktestConfigError(
                "capital_mode must be 'fixed_notional' or 'compounding'"
            )
        for name, value in (
            ("initial_capital", self.initial_capital),
            ("fixed_notional", self.fixed_notional),
            ("nav_base", self.nav_base),
        ):
            if not math.isfinite(value) or value <= 0:
                raise BacktestConfigError(f"{name} must be finite and positive")
        if self.default_buy_lot_size <= 0:
            raise BacktestConfigError("default_buy_lot_size must be positive")
        if self.settlement_sessions < 0:
            raise BacktestConfigError("settlement_sessions must be nonnegative")


@dataclass(frozen=True, slots=True)
class AccountStateCheckpoint:
    """Complete resumable account state captured after a market close."""

    time: date
    cash: float
    units: float
    pending_withdrawal: float
    positions: pl.DataFrame
    unsettled: pl.DataFrame
    entitlements: pl.DataFrame
    cash_receivables: pl.DataFrame
    stock_receivables: pl.DataFrame
    latest_target: pl.DataFrame
    pending_target_positions: pl.DataFrame
    target_revision_time: date | None = None


@dataclass(frozen=True, slots=True)
class AccountDecisionContext:
    """Causal account state exposed once at a decision close."""

    decision_date: date
    execution_date: date
    equity: float
    sizing_notional: float
    reference_weights: pl.DataFrame
    close_prices: pl.DataFrame
    checkpoint: AccountStateCheckpoint


AccountTargetProvider = Callable[[AccountDecisionContext], pl.DataFrame]


_ACCOUNT_ROW_BATCH_SIZE = 8192


class _AccountRows:
    """Bound Python output rows while retaining complete columnar results."""

    __slots__ = ("_buffer", "_chunks")

    def __init__(self) -> None:
        self._buffer: list[dict[str, Any]] = []
        self._chunks: list[pl.DataFrame] = []

    def append(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= _ACCOUNT_ROW_BATCH_SIZE:
            self._flush()

    def _flush(self) -> None:
        if self._buffer:
            self._chunks.append(pl.DataFrame(self._buffer, infer_schema_length=None))
            self._buffer.clear()

    def frame(self, *, empty_schema=None) -> pl.DataFrame:
        self._flush()
        if not self._chunks:
            return pl.DataFrame(schema=empty_schema)
        result = pl.concat(self._chunks, how="diagonal_relaxed", rechunk=False)
        self._chunks.clear()
        return result


type _OutputRows = list[dict[str, Any]] | _AccountRows


def _validate_market_prices(frame: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise InputValidationError("market_prices must be a polars DataFrame")
    required = {TIME, ASSET_ID, "open", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError(f"market_prices is missing columns: {missing}")
    result = frame.select(TIME, ASSET_ID, "open", "close").with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("open").cast(pl.Float64, strict=False),
        pl.col("close").cast(pl.Float64, strict=False),
    )
    if result.select(pl.struct(TIME, ASSET_ID).is_duplicated().any()).item():
        raise InputValidationError("market_prices must be unique by (time, asset_id)")
    invalid = result.filter(
        (
            pl.col("open").is_not_null()
            & (~pl.col("open").is_finite() | (pl.col("open") <= 0))
        )
        | (
            pl.col("close").is_not_null()
            & (~pl.col("close").is_finite() | (pl.col("close") <= 0))
        )
    )
    if invalid.height:
        raise InputValidationError(
            "market_prices open/close must be positive and finite"
        )
    return result.sort([TIME, ASSET_ID])


def _validate_target_weights(frame: pl.DataFrame) -> pl.DataFrame:
    required = {TIME, ASSET_ID, "weight"}
    if not isinstance(frame, pl.DataFrame) or not required.issubset(frame.columns):
        raise InputValidationError(
            "target_weights requires time, asset_id, and weight columns"
        )
    result = frame.select(TIME, ASSET_ID, "weight").with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("weight").cast(pl.Float64, strict=False),
    )
    if result.select(pl.struct(TIME, ASSET_ID).is_duplicated().any()).item():
        raise InputValidationError("target_weights must be unique by (time, asset_id)")
    if result.filter(
        pl.col("weight").is_null()
        | ~pl.col("weight").is_finite()
        | (pl.col("weight") < 0)
    ).height:
        raise InputValidationError("target weights must be finite and nonnegative")
    invalid_sums = (
        result.group_by(TIME)
        .agg(pl.col("weight").sum().alias("total"))
        .filter(pl.col("total") > 1.0 + 1e-8)
    )
    if invalid_sums.height:
        raise InputValidationError("target weights must sum to at most one per date")
    return result.sort([TIME, ASSET_ID])


def _empty_target_position_plans() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "decision_date": pl.Date,
            "execution_date": pl.Date,
            ASSET_ID: pl.String,
            "target_weight": pl.Float64,
            "sizing_notional": pl.Float64,
            "decision_price": pl.Float64,
            "target_quantity": pl.Int64,
        }
    )


def _build_stateful_decision_plan(
    *,
    decision_date: date,
    execution_date: date,
    prices: dict[str, dict[str, float | None]],
    state: dict[str, Any],
    config: AccountBacktestConfig,
    lots: dict[str, int],
    target_provider: AccountTargetProvider,
    minimum_trade_notional: float = 0.0,
) -> pl.DataFrame:
    equity = _state_equity(state)
    sizing_notional = (
        config.fixed_notional if config.capital_mode == "fixed_notional" else equity
    )
    if sizing_notional <= 0:
        raise InputValidationError("decision sizing notional must be positive")
    checkpoint = _checkpoint(decision_date, state)
    close_rows = [
        {ASSET_ID: asset_id, "price": float(item["close"])}
        for asset_id, item in sorted(prices.items())
        if item.get("close") is not None
    ]
    close_prices = (
        pl.DataFrame(close_rows, schema={ASSET_ID: pl.String, "price": pl.Float64})
        if close_rows
        else pl.DataFrame(schema={ASSET_ID: pl.String, "price": pl.Float64})
    )
    reference_rows = [
        {
            TIME: decision_date,
            ASSET_ID: asset_id,
            "weight": quantity * state["marks"].get(asset_id, 0.0) / sizing_notional,
        }
        for asset_id, quantity in sorted(state["positions"].items())
        if quantity
    ]
    reference_weights = (
        pl.DataFrame(
            reference_rows,
            schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64},
        )
        if reference_rows
        else pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64}
        )
    )
    provided = target_provider(
        AccountDecisionContext(
            decision_date=decision_date,
            execution_date=execution_date,
            equity=equity,
            sizing_notional=sizing_notional,
            reference_weights=reference_weights,
            close_prices=close_prices,
            checkpoint=checkpoint,
        )
    )
    if not isinstance(provided, pl.DataFrame) or not {
        ASSET_ID,
        "weight",
    }.issubset(provided.columns):
        raise InputValidationError(
            "target_provider must return asset_id and weight columns"
        )
    target_snapshot = _validate_target_weights(
        provided.select(ASSET_ID, "weight").with_columns(
            pl.lit(decision_date).cast(pl.Date).alias(TIME)
        )
    ).drop(TIME)
    current_assets = {
        asset for asset, quantity in state["positions"].items() if quantity
    }
    target_assets = set(
        target_snapshot.filter(pl.col("weight") > 0)[ASSET_ID].to_list()
    )
    price_by_asset = {
        str(row[ASSET_ID]): float(row["price"])
        for row in close_prices.iter_rows(named=True)
    }
    missing_targets = sorted(target_assets - set(price_by_asset))
    if missing_targets:
        raise InputValidationError(
            "decision sizing requires a finite close for target assets: "
            + ", ".join(missing_targets)
        )
    # A held asset without a decision-close price stays outside the immutable
    # execution plan.  The account engine will preserve its quantity and its
    # last observed mark until a later decision can price it again.
    required_assets = sorted(target_assets | (current_assets & set(price_by_asset)))
    if not required_assets:
        return _empty_target_position_plans()
    lot_frame = pl.DataFrame(
        [
            {ASSET_ID: asset_id, "lot_size": int(lots.get(asset_id, 1))}
            for asset_id in required_assets
        ],
        schema={ASSET_ID: pl.String, "lot_size": pl.Int64},
    )
    allocation = allocate_integer_positions(
        target_snapshot.filter(pl.col(ASSET_ID).is_in(required_assets)),
        close_prices.filter(pl.col(ASSET_ID).is_in(required_assets)),
        total_notional=sizing_notional,
        lot_sizes=lot_frame,
        allow_one_lot_over_target=True,
    )
    allocated = {
        str(row[ASSET_ID]): int(row["target_quantity"])
        for row in allocation.positions.iter_rows(named=True)
    }
    weights = {
        str(row[ASSET_ID]): float(row["weight"])
        for row in target_snapshot.iter_rows(named=True)
    }
    if minimum_trade_notional > 0:
        for asset_id in required_assets:
            current = int(state["positions"].get(asset_id, 0))
            change = abs(allocated.get(asset_id, 0) - current)
            if (
                weights.get(asset_id, 0.0) > 0
                and change * price_by_asset[asset_id] < minimum_trade_notional
            ):
                allocated[asset_id] = current
    return pl.DataFrame(
        [
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                ASSET_ID: asset_id,
                "target_weight": weights.get(asset_id, 0.0),
                "sizing_notional": sizing_notional,
                "decision_price": price_by_asset[asset_id],
                "target_quantity": allocated.get(asset_id, 0),
            }
            for asset_id in required_assets
        ],
        schema=_empty_target_position_plans().schema,
    )


def _validate_corporate_action_coverage(
    frame: pl.DataFrame,
    calendar: list[date],
) -> None:
    if not isinstance(frame, pl.DataFrame) or not {TIME, "is_complete"}.issubset(
        frame.columns
    ):
        raise InputValidationError(
            "corporate_action_coverage requires time and is_complete columns"
        )
    complete = frame.select(TIME, "is_complete").with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col("is_complete").cast(pl.Boolean, strict=False),
    )
    lookup = {row[TIME]: row["is_complete"] for row in complete.iter_rows(named=True)}
    missing = [value for value in calendar if lookup.get(value) is not True]
    if missing:
        rendered = ", ".join(str(value) for value in missing[:10])
        raise InputValidationError(
            "corporate-action coverage is incomplete at: " + rendered
        )


def _validate_corporate_actions(frame: pl.DataFrame | None) -> pl.DataFrame:
    columns = {
        "action_id": pl.String,
        ASSET_ID: pl.String,
        "is_implemented": pl.Boolean,
        "record_date": pl.Date,
        "ex_date": pl.Date,
        "cash_pay_date": pl.Date,
        "share_available_date": pl.Date,
        "cash_dividend_per_share": pl.Float64,
        "stock_dividend_per_share": pl.Float64,
    }
    if frame is None:
        return pl.DataFrame(schema=columns)
    required = set(columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError(f"corporate_actions is missing columns: {missing}")
    result = (
        frame.select(*columns)
        .with_columns(
            pl.col("action_id").cast(pl.String),
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("is_implemented").cast(pl.Boolean, strict=False),
            pl.col("record_date").cast(pl.Date, strict=False),
            pl.col("ex_date").cast(pl.Date, strict=False),
            pl.col("cash_pay_date").cast(pl.Date, strict=False),
            pl.col("share_available_date").cast(pl.Date, strict=False),
            pl.col("cash_dividend_per_share")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0),
            pl.col("stock_dividend_per_share")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0),
        )
        .filter(pl.col("is_implemented"))
    )
    incomplete = result.filter(
        pl.col("record_date").is_null()
        | pl.col("ex_date").is_null()
        | ((pl.col("cash_dividend_per_share") > 0) & pl.col("cash_pay_date").is_null())
        | (
            (pl.col("stock_dividend_per_share") > 0)
            & pl.col("share_available_date").is_null()
        )
    )
    if incomplete.height:
        raise InputValidationError(
            "implemented corporate actions require complete applicable dates"
        )
    invalid = result.filter(
        pl.col("action_id").is_null()
        | (pl.col("action_id").str.len_chars() == 0)
        | pl.col(ASSET_ID).is_null()
        | (pl.col(ASSET_ID).str.len_chars() == 0)
        | ~pl.col("cash_dividend_per_share").is_finite()
        | ~pl.col("stock_dividend_per_share").is_finite()
        | (pl.col("cash_dividend_per_share") < 0)
        | (pl.col("stock_dividend_per_share") < 0)
        | (pl.col("record_date") >= pl.col("ex_date"))
        | (pl.col("cash_pay_date") < pl.col("ex_date")).fill_null(False)
        | (pl.col("share_available_date") < pl.col("ex_date")).fill_null(False)
    )
    if invalid.height:
        raise InputValidationError(
            "corporate actions need finite nonnegative amounts and causal dates"
        )
    if result.select(pl.col("action_id").is_duplicated().any()).item():
        raise InputValidationError("corporate action_id values must be unique")
    return result.sort(["ex_date", ASSET_ID, "action_id"])


def _restore_state(
    config: AccountBacktestConfig,
    *,
    initial_positions: pl.DataFrame | None,
    initial_cash: float | None,
    checkpoint: AccountStateCheckpoint | None,
) -> dict[str, Any]:
    if checkpoint is not None:
        positions = {
            row[ASSET_ID]: int(row["quantity"])
            for row in checkpoint.positions.iter_rows(named=True)
        }
        available = {
            row[ASSET_ID]: int(row["available_quantity"])
            for row in checkpoint.positions.iter_rows(named=True)
        }
        marks = {
            row[ASSET_ID]: float(row["last_mark"])
            for row in checkpoint.positions.iter_rows(named=True)
            if row["last_mark"] is not None
        }
        return {
            "cash": checkpoint.cash,
            "units": checkpoint.units,
            "pending_withdrawal": checkpoint.pending_withdrawal,
            "positions": positions,
            "available": available,
            "marks": marks,
            "unsettled": checkpoint.unsettled.to_dicts(),
            "entitlements": {
                row["action_id"]: int(row["quantity"])
                for row in checkpoint.entitlements.iter_rows(named=True)
            },
            "cash_receivables": checkpoint.cash_receivables.to_dicts(),
            "stock_receivables": checkpoint.stock_receivables.to_dicts(),
            "latest_target": {
                row[ASSET_ID]: float(row["weight"])
                for row in checkpoint.latest_target.iter_rows(named=True)
            },
            "pending_target_positions": {
                row[ASSET_ID]: (
                    None
                    if row["target_quantity"] is None
                    else int(row["target_quantity"])
                )
                for row in checkpoint.pending_target_positions.iter_rows(named=True)
            },
            "target_revision_time": checkpoint.target_revision_time or checkpoint.time,
        }
    cash = config.initial_capital if initial_cash is None else float(initial_cash)
    if not math.isfinite(cash) or cash < 0:
        raise InputValidationError("initial_cash must be finite and nonnegative")
    positions: dict[str, int] = {}
    available: dict[str, int] = {}
    marks: dict[str, float] = {}
    if initial_positions is not None:
        required = {ASSET_ID, "quantity", "available_quantity", "last_mark"}
        if not required.issubset(initial_positions.columns):
            raise InputValidationError(
                "initial_positions requires asset_id, quantity, "
                "available_quantity, and last_mark"
            )
        for row in initial_positions.iter_rows(named=True):
            asset_id = str(row[ASSET_ID])
            quantity = int(row["quantity"])
            sellable = int(row["available_quantity"])
            mark = float(row["last_mark"])
            if quantity < 0 or sellable < 0 or sellable > quantity or mark <= 0:
                raise InputValidationError("initial position values are invalid")
            positions[asset_id] = quantity
            available[asset_id] = sellable
            marks[asset_id] = mark
    equity = cash + sum(positions[a] * marks[a] for a in positions)
    if equity <= 0:
        raise InputValidationError("initial account equity must be positive")
    return {
        "cash": cash,
        "units": equity / config.nav_base,
        "pending_withdrawal": 0.0,
        "positions": positions,
        "available": available,
        "marks": marks,
        "unsettled": [],
        "entitlements": {},
        "cash_receivables": [],
        "stock_receivables": [],
        "latest_target": {},
        "pending_target_positions": {},
        "target_revision_time": None,
    }


def _apply_fill(
    session: date,
    asset_id: str,
    side: Literal["buy", "sell"],
    quantity: int,
    open_price: float,
    state: dict[str, Any],
    config: AccountBacktestConfig,
    calendar: list[date],
    session_index: int,
    fill_rows: _OutputRows,
    order_id: str,
) -> float:
    rate = config.transaction_cost.slippage_for(side)
    fill_price = open_price * (1 + rate if side == "buy" else 1 - rate)
    notional = quantity * fill_price
    commission = max(
        config.transaction_cost.min_fee, notional * config.transaction_cost.rate
    )
    stamp_tax = (
        notional * config.transaction_cost.stamp_tax_rate if side == "sell" else 0.0
    )
    transfer_fee = notional * config.transaction_cost.transfer_fee_rate
    explicit_cost = commission + stamp_tax + transfer_fee
    if side == "buy":
        cash_change = -(notional + explicit_cost)
        if state["cash"] + cash_change < -1e-8:
            raise AssertionError("buy fill exceeded reserved cash")
        state["positions"][asset_id] = state["positions"].get(asset_id, 0) + quantity
        if config.settlement_sessions == 0:
            state["available"][asset_id] = (
                state["available"].get(asset_id, 0) + quantity
            )
        else:
            available_index = min(
                session_index + config.settlement_sessions,
                len(calendar) - 1,
            )
            state["unsettled"].append(
                {
                    ASSET_ID: asset_id,
                    "available_date": calendar[available_index],
                    "quantity": quantity,
                }
            )
    else:
        cash_change = notional - explicit_cost
        state["positions"][asset_id] -= quantity
        state["available"][asset_id] -= quantity
    state["cash"] += cash_change
    fill_rows.append(
        {
            TIME: session,
            "order_id": order_id,
            ASSET_ID: asset_id,
            "side": side,
            "quantity": quantity,
            "open_price": open_price,
            "fill_price": fill_price,
            "notional": notional,
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
            "slippage_cost": quantity * abs(fill_price - open_price),
            "cash_change": cash_change,
        }
    )
    return explicit_cost + quantity * abs(fill_price - open_price)


def _apply_corporate_actions(
    state: dict[str, Any],
    session: date,
    *,
    actions_by_ex: dict[date, tuple[dict[str, Any], ...]],
    actions_by_pay: dict[date, tuple[dict[str, Any], ...]],
    actions_by_list: dict[date, tuple[dict[str, Any], ...]],
    receivable_rows: _OutputRows,
) -> None:
    for action in actions_by_ex.get(session, ()):
        entitlement = state["entitlements"].get(action["action_id"], 0)
        cash_amount = entitlement * action["cash_dividend_per_share"]
        stock_quantity = int(entitlement * action["stock_dividend_per_share"])
        if cash_amount:
            state["cash_receivables"].append(
                {
                    "action_id": action["action_id"],
                    ASSET_ID: action[ASSET_ID],
                    "pay_date": action["cash_pay_date"],
                    "amount": cash_amount,
                }
            )
            receivable_rows.append(
                {
                    TIME: session,
                    "action_id": action["action_id"],
                    ASSET_ID: action[ASSET_ID],
                    "kind": "cash_created",
                    "amount": cash_amount,
                    "quantity": None,
                }
            )
        if stock_quantity:
            state["stock_receivables"].append(
                {
                    "action_id": action["action_id"],
                    ASSET_ID: action[ASSET_ID],
                    "available_date": action["share_available_date"],
                    "quantity": stock_quantity,
                }
            )
            receivable_rows.append(
                {
                    TIME: session,
                    "action_id": action["action_id"],
                    ASSET_ID: action[ASSET_ID],
                    "kind": "stock_created",
                    "amount": None,
                    "quantity": stock_quantity,
                }
            )
    pay_ids = {action["action_id"] for action in actions_by_pay.get(session, ())}
    retained_cash = []
    for receivable in state["cash_receivables"]:
        if receivable["action_id"] in pay_ids:
            state["cash"] += receivable["amount"]
            receivable_rows.append(
                {
                    TIME: session,
                    "action_id": receivable["action_id"],
                    ASSET_ID: receivable[ASSET_ID],
                    "kind": "cash_paid",
                    "amount": receivable["amount"],
                    "quantity": None,
                }
            )
        else:
            retained_cash.append(receivable)
    state["cash_receivables"] = retained_cash
    list_ids = {action["action_id"] for action in actions_by_list.get(session, ())}
    retained_stock = []
    for receivable in state["stock_receivables"]:
        if receivable["action_id"] in list_ids:
            asset_id = receivable[ASSET_ID]
            quantity = receivable["quantity"]
            state["positions"][asset_id] = (
                state["positions"].get(asset_id, 0) + quantity
            )
            state["available"][asset_id] = (
                state["available"].get(asset_id, 0) + quantity
            )
            receivable_rows.append(
                {
                    TIME: session,
                    "action_id": receivable["action_id"],
                    ASSET_ID: asset_id,
                    "kind": "stock_available",
                    "amount": None,
                    "quantity": quantity,
                }
            )
        else:
            retained_stock.append(receivable)
    state["stock_receivables"] = retained_stock


def _record_entitlements(
    session: date,
    state: dict[str, Any],
    actions: tuple[dict[str, Any], ...],
) -> None:
    for action in actions:
        state["entitlements"][action["action_id"]] = state["positions"].get(
            action[ASSET_ID], 0
        )


def _checkpoint(session: date, state: dict[str, Any]) -> AccountStateCheckpoint:
    position_rows = [
        {
            ASSET_ID: asset_id,
            "quantity": quantity,
            "available_quantity": state["available"].get(asset_id, 0),
            "last_mark": state["marks"].get(asset_id),
        }
        for asset_id, quantity in sorted(state["positions"].items())
    ]
    entitlement_rows = [
        {"action_id": action_id, "quantity": quantity}
        for action_id, quantity in sorted(state["entitlements"].items())
    ]
    target_rows = [
        {ASSET_ID: asset_id, "weight": weight}
        for asset_id, weight in sorted(state["latest_target"].items())
    ]
    pending_target_rows = [
        {ASSET_ID: asset_id, "target_quantity": quantity}
        for asset_id, quantity in sorted(state["pending_target_positions"].items())
    ]
    return AccountStateCheckpoint(
        time=session,
        cash=state["cash"],
        units=state["units"],
        pending_withdrawal=state["pending_withdrawal"],
        positions=_rows_frame(position_rows),
        unsettled=_rows_frame(state["unsettled"]),
        entitlements=_rows_frame(entitlement_rows),
        cash_receivables=_rows_frame(state["cash_receivables"]),
        stock_receivables=_rows_frame(state["stock_receivables"]),
        latest_target=_rows_frame(target_rows),
        pending_target_positions=_rows_frame(pending_target_rows),
        target_revision_time=state["target_revision_time"],
    )


def _release_settled_shares(state: dict[str, Any], session: date) -> None:
    remaining = []
    for item in state["unsettled"]:
        if item["available_date"] <= session:
            asset_id = item[ASSET_ID]
            state["available"][asset_id] = (
                state["available"].get(asset_id, 0) + item["quantity"]
            )
        else:
            remaining.append(item)
    state["unsettled"] = remaining


def _mark_positions(
    state: dict[str, Any],
    prices: dict[str, dict[str, float | None]],
    *,
    use: Literal["open", "close"],
) -> None:
    for asset_id in state["positions"]:
        mark = prices.get(asset_id, {}).get(use)
        if mark is not None:
            state["marks"][asset_id] = float(mark)


def _state_equity(state: dict[str, Any]) -> float:
    return (
        state["cash"]
        + _position_value(state)
        + _cash_receivable_value(state)
        + _stock_receivable_value(state)
    )


def _position_value(state: dict[str, Any]) -> float:
    return sum(
        quantity * state["marks"].get(asset_id, 0.0)
        for asset_id, quantity in state["positions"].items()
    )


def _cash_receivable_value(state: dict[str, Any]) -> float:
    return sum(float(item["amount"]) for item in state["cash_receivables"])


def _stock_receivable_value(state: dict[str, Any]) -> float:
    return sum(
        item["quantity"] * state["marks"].get(item[ASSET_ID], 0.0)
        for item in state["stock_receivables"]
    )


def _price_lookup(
    frame: pl.DataFrame,
) -> dict[date, dict[str, dict[str, float | None]]]:
    result: dict[date, dict[str, dict[str, float | None]]] = {}
    for row in frame.iter_rows(named=True):
        result.setdefault(row[TIME], {})[row[ASSET_ID]] = {
            "open": row["open"],
            "close": row["close"],
        }
    return result


def _group_actions(
    frame: pl.DataFrame,
    column: str,
) -> dict[date, tuple[dict[str, Any], ...]]:
    grouped: dict[date, list[dict[str, Any]]] = {}
    for row in frame.filter(pl.col(column).is_not_null()).iter_rows(named=True):
        grouped.setdefault(row[column], []).append(row)
    return {
        key: tuple(sorted(value, key=lambda item: (item[ASSET_ID], item["action_id"])))
        for key, value in grouped.items()
    }


def _availability_lookup(
    frame: pl.DataFrame | None,
) -> dict[tuple[date, str], tuple[bool, bool, str]]:
    if frame is None:
        return {}
    required = {TIME, ASSET_ID, "can_buy", "can_sell", "reason"}
    if not required.issubset(frame.columns):
        raise InputValidationError(
            "execution_availability is missing: "
            f"{sorted(required - set(frame.columns))}"
        )
    normalized = frame.select(*required).with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("can_buy").cast(pl.Boolean, strict=False),
        pl.col("can_sell").cast(pl.Boolean, strict=False),
        pl.col("reason").cast(pl.String, strict=False).fill_null(""),
    )
    return {
        (row[TIME], row[ASSET_ID]): (row["can_buy"], row["can_sell"], row["reason"])
        for row in normalized.iter_rows(named=True)
    }


def _lot_size_lookup(frame: pl.DataFrame | None, default: int) -> dict[str, int]:
    if frame is None:
        return {}
    if not {ASSET_ID, "buy_lot_size"}.issubset(frame.columns):
        raise InputValidationError("lot_sizes requires asset_id and buy_lot_size")
    result = {}
    for row in frame.iter_rows(named=True):
        size = int(row["buy_lot_size"])
        if size <= 0:
            raise InputValidationError("buy_lot_size must be positive")
        result[str(row[ASSET_ID])] = size
    return result


def _rows_frame(rows: _OutputRows, *, empty_schema=None) -> pl.DataFrame:
    if isinstance(rows, _AccountRows):
        return rows.frame(empty_schema=empty_schema)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        if rows
        else pl.DataFrame(schema=empty_schema)
    )


def complete_checkpoint_entitlements(
    checkpoint, actions, *, historical_positions=None, historical_sessions=None
):
    """Recover record-date quantities for newly matured corporate-action facts.

    Sparse held positions and the complete historical session inventory are
    required when an action was not yet present in the checkpoint's input.
    Current holdings cannot stand in for holdings on a past record date.
    """
    actions = _validate_corporate_actions(actions)
    known = (
        set(checkpoint.entitlements.get_column("action_id").to_list())
        if "action_id" in checkpoint.entitlements.columns
        else set()
    )
    missing = actions.filter(
        (pl.col("record_date") <= checkpoint.time)
        & ~pl.col("action_id").is_in(sorted(known))
    )
    if missing.is_empty():
        return checkpoint
    if historical_positions is None or historical_sessions is None:
        raise InputValidationError(
            "checkpoint requires verified record-date position history "
            "for newly matured corporate actions"
        )
    sessions = set(historical_sessions.get_column(TIME).to_list())
    if checkpoint.time not in sessions:
        raise InputValidationError(
            "verified account history must include the checkpoint session"
        )
    rows = []
    for action in missing.iter_rows(named=True):
        if action["record_date"] not in sessions:
            # Match full simulation: dates before account creation or without
            # an account session never registered a share entitlement.
            rows.append({"action_id": action["action_id"], "quantity": 0})
            continue
        holding = historical_positions.filter(
            (pl.col(TIME) == action["record_date"])
            & (pl.col(ASSET_ID) == action[ASSET_ID])
        )
        rows.append(
            {
                "action_id": action["action_id"],
                "quantity": int(holding["quantity"][0]) if holding.height else 0,
            }
        )
    added = pl.DataFrame(rows, schema={"action_id": pl.String, "quantity": pl.Int64})
    entitlements = (
        added
        if checkpoint.entitlements.is_empty()
        else pl.concat([checkpoint.entitlements, added], how="vertical_relaxed")
    )
    return replace(checkpoint, entitlements=entitlements.sort("action_id"))
