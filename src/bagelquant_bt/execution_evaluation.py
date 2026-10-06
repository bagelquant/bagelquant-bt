"""Signed-share execution evaluation and its causal weight-planning bridge.

The account module owns cash, settlement, fills, marks and corporate actions.
This module composes those primitives with immutable delta orders, a fee-free
reference account and FIFO attribution. Prices and timing always come from the
caller; no exchange-specific convention is selected here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import date
from typing import Any, Literal

import polars as pl

from . import account as _account
from .account import AccountBacktestConfig, AccountStateCheckpoint
from .config import TransactionCostConfig
from .exceptions import InputValidationError


@dataclass(frozen=True, slots=True)
class ExecutionCostQuote:
    """Money fees and the explicit fill price for one aggregate daily fill."""

    fee: float
    fill_price: float


@dataclass(frozen=True, slots=True)
class ExecutionCostRule:
    """An identified, pure rule whose total buy charge increases with quantity.

    ``quote(side, quantity, reference_price, execution_date, asset_id)`` returns
    an ExecutionCostQuote. The ID, version and JSON parameters are numerical
    identity; the callable's repr or source is never an identity substitute.
    """

    rule_id: str
    version: str
    parameters: Mapping[str, Any]
    quote: Callable[[str, int, float, date, str], ExecutionCostQuote] = field(
        compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.version.strip():
            raise InputValidationError("custom cost rules need stable ID and version")
        if not callable(self.quote):
            raise InputValidationError("custom cost quote must be callable")
        json.dumps(dict(self.parameters), sort_keys=True, allow_nan=False)


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Explicit neutral account and money-cost settings."""

    rate: float = 0.0005
    min_fee: float = 5.0
    buy_slippage_rate: float = 0.0
    sell_slippage_rate: float = 0.0
    stamp_tax_rate: float = 0.0
    transfer_fee_rate: float = 0.0
    default_buy_lot_size: int = 1
    settlement_sessions: int = 0
    retry_unfilled: bool = False
    annualization: int = 252
    cost_rule: ExecutionCostRule | None = None
    capital_mode: Literal["compounding", "fixed_notional"] = "compounding"
    fixed_notional: float | None = None

    def __post_init__(self) -> None:
        _transaction_cost(self)
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (
                self.default_buy_lot_size,
                self.settlement_sessions,
                self.annualization,
            )
        ):
            raise InputValidationError(
                "lot, settlement and annualization must be integers"
            )
        if self.default_buy_lot_size < 1 or self.settlement_sessions < 0:
            raise InputValidationError(
                "lot size must be positive; settlement nonnegative"
            )
        if self.annualization < 1:
            raise InputValidationError("annualization must be positive")
        if self.capital_mode not in {"compounding", "fixed_notional"}:
            raise InputValidationError("invalid capital_mode")
        if self.fixed_notional is not None and (
            not math.isfinite(self.fixed_notional) or self.fixed_notional <= 0
        ):
            raise InputValidationError("fixed_notional must be finite and positive")

    def identity(self) -> dict[str, Any]:
        """Return serializable numerical settings, excluding executable objects."""
        result = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "cost_rule"
        }
        if self.cost_rule is not None:
            result["cost_rule"] = {
                "rule_id": self.cost_rule.rule_id,
                "version": self.cost_rule.version,
                "parameters": dict(self.cost_rule.parameters),
            }
        return result


@dataclass(frozen=True, slots=True)
class ExecutionStateCheckpoint:
    """Both account states, FIFO state and frozen future/retry deltas."""

    time: date
    actual: AccountStateCheckpoint
    ideal: AccountStateCheckpoint
    actual_lots: pl.DataFrame
    ideal_lots: pl.DataFrame
    pending_transactions: pl.DataFrame
    processed_plan_ids: tuple[str, ...]
    actual_realized_gross: float
    actual_realized_net: float
    ideal_realized_gross: float
    actual_income: float
    ideal_income: float
    fee_total: float
    initial_capital: float
    prefix_hash: str
    entitlement_lots: pl.DataFrame
    corporate_pnl: pl.DataFrame


@dataclass(frozen=True, slots=True)
class ExecutionEvaluationResult:
    """Inspectable account, order, FIFO and reference-result tables."""

    tables: dict[str, pl.DataFrame]
    metrics: dict[str, Any]
    checkpoint: ExecutionStateCheckpoint


_PLAN_SCHEMA = {
    "plan_id": pl.String,
    "execution_date": pl.Date,
    "asset_id": pl.String,
    "quantity": pl.Int64,
    "decision_date": pl.Date,
    "reference_price": pl.Float64,
}
_PENDING_SCHEMA = {**_PLAN_SCHEMA, "_retry": pl.Boolean}
_LOT_SCHEMA = {
    "plan_id": pl.String,
    "asset_id": pl.String,
    "entry_date": pl.Date,
    "quantity": pl.Int64,
    "price": pl.Float64,
    "entry_fee": pl.Float64,
}
_FILL_SCHEMA = {
    "time": pl.Date,
    "order_id": pl.String,
    "asset_id": pl.String,
    "side": pl.String,
    "quantity": pl.Int64,
    **{
        name: pl.Float64
        for name in (
            "open_price",
            "fill_price",
            "notional",
            "commission",
            "stamp_tax",
            "transfer_fee",
            "slippage_cost",
            "cash_change",
            "explicit_cost",
        )
    },
    "plan_id": pl.String,
    "execution_date": pl.Date,
}
_ORDER_SCHEMA = {
    "time": pl.Date,
    "execution_date": pl.Date,
    "plan_id": pl.String,
    "order_id": pl.String,
    "asset_id": pl.String,
    "side": pl.String,
    "requested_quantity": pl.Int64,
    "filled_quantity": pl.Int64,
    "order_quantity": pl.Int64,
    "unfilled_quantity": pl.Int64,
    "status": pl.String,
    "reason": pl.String,
    "expires_at": pl.Date,
}
_MATCH_SCHEMA = {
    "ledger": pl.String,
    "asset_id": pl.String,
    "buy_plan_id": pl.String,
    "sell_plan_id": pl.String,
    "entry_date": pl.Date,
    "exit_date": pl.Date,
    "quantity": pl.Int64,
    **{
        name: pl.Float64
        for name in (
            "entry_price",
            "exit_price",
            "gross_pnl",
            "entry_cost",
            "exit_cost",
            "net_pnl",
        )
    },
}
_PNL_SCHEMA = {
    "plan_id": pl.String,
    "asset_id": pl.String,
    "side": pl.String,
    **{f"{ledger}_quantity": pl.Int64 for ledger in ("target", "actual")},
    **{
        f"{ledger}_{name}": pl.Float64
        for ledger in ("target", "actual")
        for name in (
            "notional",
            "cost",
            "realized_pnl",
            "unrealized_pnl",
            "pnl",
            "return",
            "dividend_income",
        )
    },
    **{f"{ledger}_return_reason": pl.String for ledger in ("target", "actual")},
}
_ENTITLEMENT_SCHEMA = {
    "action_id": pl.String,
    "ledger": pl.String,
    "plan_id": pl.String,
    "asset_id": pl.String,
    "quantity": pl.Int64,
}
_CORPORATE_PNL_SCHEMA = {
    "time": pl.Date,
    "action_id": pl.String,
    "ledger": pl.String,
    "plan_id": pl.String,
    "asset_id": pl.String,
    "cash_income": pl.Float64,
    "stock_quantity": pl.Int64,
    "share_available_date": pl.Date,
}
_LOT_HISTORY_SCHEMA = {"time": pl.Date, **_LOT_SCHEMA, "mark_price": pl.Float64}


def _frame(rows, schema=None) -> pl.DataFrame:
    if isinstance(rows, _account._AccountRows):
        return rows.frame(empty_schema=schema)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        if rows
        else pl.DataFrame(schema=schema)
    )


def _transaction_cost(config: ExecutionConfig) -> TransactionCostConfig:
    return TransactionCostConfig(
        **{
            name: getattr(config, name)
            for name in (
                "rate",
                "min_fee",
                "buy_slippage_rate",
                "sell_slippage_rate",
                "stamp_tax_rate",
                "transfer_fee_rate",
            )
        }
    )


def _account_config(config: ExecutionConfig, capital: float, *, ideal=False):
    return AccountBacktestConfig(
        capital_mode=config.capital_mode,
        initial_capital=capital,
        fixed_notional=config.fixed_notional or capital,
        default_buy_lot_size=1 if ideal else config.default_buy_lot_size,
        settlement_sessions=0 if ideal else config.settlement_sessions,
        retry_blocked_orders=False,
        transaction_cost=(
            TransactionCostConfig(
                rate=0,
                min_fee=0,
                buy_slippage_rate=0,
                sell_slippage_rate=0,
                stamp_tax_rate=0,
                transfer_fee_rate=0,
            )
            if ideal
            else _transaction_cost(config)
        ),
    )


def _plans(frame: pl.DataFrame) -> pl.DataFrame:
    if "execution_date" not in frame.columns and "time" in frame.columns:
        frame = frame.rename({"time": "execution_date"})
    required = {"plan_id", "execution_date", "asset_id", "quantity"}
    if not required <= set(frame.columns):
        raise InputValidationError(f"transactions require {sorted(required)}")
    if "decision_date" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Date).alias("decision_date"))
    if "reference_price" not in frame.columns:
        frame = frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("reference_price")
        )
    original = frame.get_column("quantity").cast(pl.Float64, strict=False)
    if original.is_null().any() or (original != original.floor()).any():
        raise InputValidationError("transaction quantity must be an integer")
    result = frame.select(list(_PLAN_SCHEMA)).cast(_PLAN_SCHEMA, strict=False)
    if result.filter(
        pl.col("plan_id").is_null()
        | (pl.col("plan_id").str.len_chars() == 0)
        | pl.col("asset_id").is_null()
        | (pl.col("asset_id").str.len_chars() == 0)
        | pl.col("execution_date").is_null()
        | pl.col("quantity").is_null()
        | (pl.col("quantity") == 0)
        | (pl.col("decision_date") >= pl.col("execution_date")).fill_null(False)
        | (
            pl.col("reference_price").is_not_null()
            & (
                ~pl.col("reference_price").is_finite()
                | (pl.col("reference_price") <= 0)
            )
        )
    ).height:
        raise InputValidationError("transactions contain invalid dates/IDs/quantities")
    if result.get_column("plan_id").is_duplicated().any():
        raise InputValidationError("plan_id must be unique; orders are never netted")
    return result.sort("execution_date", "asset_id", "plan_id")


def _market(frame: pl.DataFrame) -> pl.DataFrame:
    rename = {}
    if "execution_price" in frame.columns:
        if "open" in frame.columns:
            raise InputValidationError("provide execution_price or open, not both")
        rename["execution_price"] = "open"
    if "valuation_price" in frame.columns:
        if "close" in frame.columns:
            raise InputValidationError("provide valuation_price or close, not both")
        rename["valuation_price"] = "close"
    normalized = frame.rename(rename)
    result = _account._validate_market_prices(normalized)
    if "reference_price" in normalized.columns:
        reference = normalized.select(
            "time", "asset_id", pl.col("reference_price").cast(pl.Float64, strict=False)
        )
        if reference.filter(
            pl.col("reference_price").is_not_null()
            & (
                ~pl.col("reference_price").is_finite()
                | (pl.col("reference_price") <= 0)
            )
        ).height:
            raise InputValidationError("reference prices must be positive and finite")
        result = result.join(reference, on=["time", "asset_id"], how="left")
    if result.filter(
        pl.col("time").is_null()
        | pl.col("asset_id").is_null()
        | (pl.col("asset_id").str.len_chars() == 0)
    ).height:
        raise InputValidationError("market prices require valid time and asset_id")
    return result


def _quote(config, side, quantity, price, session, asset):
    if quantity == 0:
        return ExecutionCostQuote(0.0, price)
    if config.cost_rule is not None:
        quote = config.cost_rule.quote(side, quantity, price, session, asset)
        if not isinstance(quote, ExecutionCostQuote):
            raise InputValidationError(
                "custom cost rule must return ExecutionCostQuote"
            )
    else:
        cost = _transaction_cost(config)
        fill = price * (
            1 + cost.slippage_for(side)
            if side == "buy"
            else 1 - cost.slippage_for(side)
        )
        amount = quantity * fill
        fee = max(amount * cost.rate, cost.min_fee) + amount * (
            cost.transfer_fee_rate + (cost.stamp_tax_rate if side == "sell" else 0)
        )
        quote = ExecutionCostQuote(fee, fill)
    if (
        not math.isfinite(quote.fee)
        or quote.fee < 0
        or not math.isfinite(quote.fill_price)
        or quote.fill_price <= 0
    ):
        raise InputValidationError(
            "cost quote needs nonnegative finite fee/positive price"
        )
    if (side == "buy" and quote.fill_price < price) or (
        side == "sell" and quote.fill_price > price
    ):
        raise InputValidationError("slippage must not improve the reference price")
    return quote


def _fill(state, row, qty, price, session, index, sessions, config, fills, *, ideal):
    side = "buy" if row["quantity"] > 0 else "sell"
    if (
        not ideal
        and side == "buy"
        and index + config.settlement_sessions >= len(sessions)
    ):
        raise InputValidationError(
            "buy settlement requires known future trading sessions"
        )
    quote = (
        ExecutionCostQuote(0.0, price)
        if ideal
        else _quote(config, side, qty, price, session, row["asset_id"])
    )
    base = _account_config(config, state["units"], ideal=ideal)
    slippage = abs(quote.fill_price / price - 1)
    cost = TransactionCostConfig(
        rate=quote.fee / (qty * quote.fill_price),
        min_fee=0,
        buy_slippage_rate=slippage if side == "buy" else 0,
        sell_slippage_rate=slippage if side == "sell" else 0,
        stamp_tax_rate=0,
        transfer_fee_rate=0,
    )
    fill_rows = []
    _account._apply_fill(
        session,
        row["asset_id"],
        side,
        qty,
        price,
        state,
        replace(base, transaction_cost=cost),
        sessions,
        index,
        fill_rows,
        row["plan_id"],
    )
    fill = fill_rows[-1]
    fill.update(plan_id=row["plan_id"], execution_date=session, explicit_cost=quote.fee)
    state["marks"].setdefault(row["asset_id"], price)
    if not ideal and config.cost_rule is None:
        notional = fill["notional"]
        fill.update(
            commission=max(config.min_fee, config.rate * notional),
            stamp_tax=notional * config.stamp_tax_rate if side == "sell" else 0.0,
            transfer_fee=notional * config.transfer_fee_rate,
        )
    fills.append(fill)
    return fill


def _fifo_fill(lots, fill, matches, ledger):
    asset = fill["asset_id"]
    if fill["side"] == "buy":
        lots.append(
            {
                "plan_id": fill["plan_id"],
                "asset_id": asset,
                "entry_date": fill["time"],
                "quantity": fill["quantity"],
                "price": fill["fill_price"],
                "entry_fee": fill["explicit_cost"],
            }
        )
        return 0.0, 0.0
    remaining = fill["quantity"]
    gross = net = 0.0
    for lot in lots:
        if lot["asset_id"] != asset or lot["quantity"] == 0:
            continue
        quantity = min(remaining, lot["quantity"])
        entry_fee = lot["entry_fee"] * quantity / lot["quantity"]
        exit_fee = fill["explicit_cost"] * quantity / fill["quantity"]
        pnl = quantity * (fill["fill_price"] - lot["price"])
        matches.append(
            {
                "ledger": ledger,
                "asset_id": asset,
                "buy_plan_id": lot["plan_id"],
                "sell_plan_id": fill["plan_id"],
                "entry_date": lot["entry_date"],
                "exit_date": fill["time"],
                "quantity": quantity,
                "entry_price": lot["price"],
                "exit_price": fill["fill_price"],
                "gross_pnl": pnl,
                "entry_cost": entry_fee,
                "exit_cost": exit_fee,
                "net_pnl": pnl - entry_fee - exit_fee,
            }
        )
        lot["quantity"] -= quantity
        lot["entry_fee"] -= entry_fee
        remaining -= quantity
        gross += pnl
        net += pnl - entry_fee - exit_fee
        if remaining == 0:
            break
    if remaining:
        raise AssertionError("FIFO inventory does not match account inventory")
    return gross, net


def _unrealized(lots, state):
    return sum(
        lot["quantity"] * (state["marks"][lot["asset_id"]] - lot["price"])
        for lot in lots
        if lot["quantity"]
    )


def _corporate_lots(lots, rows, allocations, *, session, ledger):
    """Retain the originating transaction for distributed zero-cost shares."""
    for row in rows:
        if row["kind"] == "stock_available":
            for allocation in allocations:
                if (
                    allocation["action_id"] == row["action_id"]
                    and allocation["ledger"] == ledger
                    and allocation["stock_quantity"]
                ):
                    lots.append(
                        {
                            "plan_id": allocation["plan_id"],
                            "asset_id": row["asset_id"],
                            "entry_date": session,
                            "quantity": allocation["stock_quantity"],
                            "price": 0.0,
                            "entry_fee": 0.0,
                        }
                    )


def _lot_entitlements(lots, action, ledger):
    quantities = {}
    for lot in lots:
        if lot["asset_id"] == action["asset_id"] and lot["quantity"]:
            quantities[lot["plan_id"]] = (
                quantities.get(lot["plan_id"], 0) + lot["quantity"]
            )
    return [
        {
            "action_id": action["action_id"],
            "ledger": ledger,
            "asset_id": action["asset_id"],
            "plan_id": plan,
            "quantity": quantity,
        }
        for plan, quantity in sorted(quantities.items())
    ]


def _corporate_allocations(action, entitlements, ledger):
    selected = [
        row
        for row in entitlements
        if row["action_id"] == action["action_id"] and row["ledger"] == ledger
    ]
    ratio = action["stock_dividend_per_share"]
    allocated = {
        row["plan_id"]: math.floor(row["quantity"] * ratio) for row in selected
    }
    total = int(sum(row["quantity"] for row in selected) * ratio)
    remainder = total - sum(allocated.values())
    ordered = sorted(
        selected,
        key=lambda row: (
            -(row["quantity"] * ratio - math.floor(row["quantity"] * ratio)),
            row["plan_id"],
        ),
    )
    for row in ordered[:remainder]:
        allocated[row["plan_id"]] += 1
    return [
        {
            "time": action["ex_date"],
            "action_id": action["action_id"],
            "ledger": ledger,
            "plan_id": row["plan_id"],
            "asset_id": action["asset_id"],
            "cash_income": row["quantity"] * action["cash_dividend_per_share"],
            "stock_quantity": allocated[row["plan_id"]],
            "share_available_date": action["share_available_date"],
        }
        for row in selected
    ]


def _prefix_hash(plans, prices, availability, actions, calendar, config, through):
    return input_prefix(
        transactions=plans,
        prices=prices,
        availability=availability,
        actions=actions,
        calendar=calendar,
        settings=config.identity(),
        through=through,
    )


def evaluate_execution(
    transactions: pl.DataFrame,
    market_prices: pl.DataFrame,
    *,
    initial_capital: float,
    config: ExecutionConfig | None = None,
    checkpoint: ExecutionStateCheckpoint | None = None,
    calendar: pl.DataFrame | None = None,
    execution_availability: pl.DataFrame | None = None,
    lot_sizes: pl.DataFrame | None = None,
    corporate_actions: pl.DataFrame | None = None,
    corporate_action_coverage: pl.DataFrame | None = None,
    available_date: date | None = None,
    historical_positions: pl.DataFrame | None = None,
    historical_ideal_positions: pl.DataFrame | None = None,
    historical_sessions: pl.DataFrame | None = None,
    historical_open_lots: pl.DataFrame | None = None,
    historical_ideal_open_lots: pl.DataFrame | None = None,
    check_canceled: Callable[[], None] = lambda: None,
    progress: Callable[[int, int], None] = lambda *_: None,
) -> ExecutionEvaluationResult:
    """Evaluate frozen signed share deltas, rejecting an infeasible ideal plan.

    Sell orders execute first, then buys, ordered by asset and stable plan ID.
    Actual fills can be reduced; fee-free reference orders execute completely.
    Historical inputs must be supplied again when resuming a checkpoint so
    its causal prefix can be verified before any continuation is returned.
    """
    return _run(
        _plans(transactions),
        _market(market_prices),
        initial_capital=initial_capital,
        config=config or ExecutionConfig(),
        checkpoint=checkpoint,
        calendar=calendar,
        execution_availability=execution_availability,
        lot_sizes=lot_sizes,
        corporate_actions=corporate_actions,
        corporate_action_coverage=corporate_action_coverage,
        available_date=available_date,
        historical_positions=historical_positions,
        historical_ideal_positions=historical_ideal_positions,
        historical_sessions=historical_sessions,
        historical_open_lots=historical_open_lots,
        historical_ideal_open_lots=historical_ideal_open_lots,
        check_canceled=check_canceled,
        progress=progress,
    )


def run_execution_from_weights(
    weights: pl.DataFrame,
    decisions: pl.DataFrame,
    market_prices: pl.DataFrame,
    *,
    initial_capital: float,
    calendar: pl.DataFrame,
    session_lag: int = 1,
    reference_price_mode: Literal["execution", "decision"] = "execution",
    config: ExecutionConfig | None = None,
    checkpoint: ExecutionStateCheckpoint | None = None,
    execution_availability: pl.DataFrame | None = None,
    lot_sizes: pl.DataFrame | None = None,
    corporate_actions: pl.DataFrame | None = None,
    corporate_action_coverage: pl.DataFrame | None = None,
    available_date: date | None = None,
    historical_positions: pl.DataFrame | None = None,
    historical_ideal_positions: pl.DataFrame | None = None,
    historical_sessions: pl.DataFrame | None = None,
    historical_open_lots: pl.DataFrame | None = None,
    historical_ideal_open_lots: pl.DataFrame | None = None,
    check_canceled: Callable[[], None] = lambda: None,
    progress: Callable[[int, int], None] = lambda *_: None,
) -> ExecutionEvaluationResult:
    """Freeze deltas at each saved rebalance close using the actual account.

    Caller-selected ``session_lag`` refers to the supplied calendar. Hold and
    unavailable decisions produce no plans; execution never re-sizes a plan.
    """
    if session_lag < 1:
        raise InputValidationError("weight decisions require an explicit positive lag")
    if reference_price_mode not in {"execution", "decision"}:
        raise InputValidationError("reference_price_mode must be execution or decision")
    value = "weight" if "weight" in weights.columns else "value"
    targets = _account._validate_target_weights(weights.rename({value: "weight"}))
    if not {"time", "status"} <= set(decisions.columns):
        raise InputValidationError("decisions require time and status")
    decision_rows = (
        decisions.select("time", "status")
        .with_columns(pl.col("time").cast(pl.Date))
        .sort("time")
    )
    if (
        decision_rows["time"].is_duplicated().any()
        or decision_rows.filter(
            ~pl.col("status").is_in(["rebalance", "hold", "unavailable"])
        ).height
    ):
        raise InputValidationError("decisions must be unique with known statuses")
    sessions = calendar.get_column("time").cast(pl.Date).unique().sort().to_list()
    schedule = {}
    for row in decision_rows.filter(pl.col("status") == "rebalance").to_dicts():
        if row["time"] not in sessions:
            raise InputValidationError(
                "decision date must be a supplied trading session"
            )
        offset = sessions.index(row["time"]) + session_lag
        if offset >= len(sessions):
            raise InputValidationError("known future execution calendar is incomplete")
        schedule[row["time"]] = sessions[offset]
    saved = {
        day[0]: frame.drop("time")
        for day, frame in targets.partition_by("time", as_dict=True).items()
    }
    config = config or ExecutionConfig()
    return _run(
        pl.DataFrame(schema=_PLAN_SCHEMA),
        _market(market_prices),
        initial_capital=initial_capital,
        config=config,
        checkpoint=checkpoint,
        calendar=calendar,
        execution_availability=execution_availability,
        lot_sizes=lot_sizes,
        corporate_actions=corporate_actions,
        corporate_action_coverage=corporate_action_coverage,
        available_date=available_date,
        weight_targets=saved,
        schedule=schedule,
        weight_evidence=targets,
        decision_evidence=decision_rows,
        reference_price_mode=reference_price_mode,
        historical_positions=historical_positions,
        historical_ideal_positions=historical_ideal_positions,
        historical_sessions=historical_sessions,
        historical_open_lots=historical_open_lots,
        historical_ideal_open_lots=historical_ideal_open_lots,
        check_canceled=check_canceled,
        progress=progress,
    )


def _run(
    plans,
    prices,
    *,
    initial_capital,
    config,
    checkpoint,
    calendar,
    execution_availability,
    lot_sizes,
    corporate_actions,
    corporate_action_coverage,
    available_date,
    weight_targets=None,
    schedule=None,
    weight_evidence=None,
    decision_evidence=None,
    historical_positions=None,
    historical_ideal_positions=None,
    historical_sessions=None,
    historical_open_lots=None,
    historical_ideal_open_lots=None,
    check_canceled=lambda: None,
    progress=lambda *_: None,
    reference_price_mode="execution",
):
    check_canceled()
    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise InputValidationError("initial_capital is required, finite and positive")
    sessions_frame = (
        prices.select("time").unique().sort("time")
        if calendar is None
        else calendar.select(pl.col("time").cast(pl.Date)).unique().sort("time")
    )
    cutoff = available_date or prices["time"].max()
    if cutoff is None:
        raise InputValidationError("execution requires market sessions")
    historical = sessions_frame.filter(pl.col("time") <= cutoff)
    sessions = historical["time"].to_list()
    all_sessions = sessions_frame["time"].to_list()
    if not set(prices["time"].to_list()) <= set(all_sessions):
        raise InputValidationError("market dates must belong to the explicit calendar")
    if not set(plans["execution_date"].to_list()) <= set(all_sessions):
        raise InputValidationError("transaction dates must belong to the calendar")
    prices = prices.filter(pl.col("time") <= cutoff)
    actions = _account._validate_corporate_actions(corporate_actions)
    if corporate_action_coverage is not None:
        _account._validate_corporate_action_coverage(
            corporate_action_coverage, sessions
        )
    availability = _account._availability_lookup(execution_availability)
    if execution_availability is not None:
        if (
            execution_availability.select(
                pl.struct("time", "asset_id").is_duplicated().any()
            ).item()
            or execution_availability.select(
                pl.any_horizontal(
                    pl.col("time", "asset_id", "can_buy", "can_sell").is_null()
                ).any()
            ).item()
        ):
            raise InputValidationError(
                "execution availability needs unique finite keys/rules"
            )
    if lot_sizes is not None:
        raw_sizes = lot_sizes["buy_lot_size"].cast(pl.Float64, strict=False)
        if (
            lot_sizes["asset_id"].is_duplicated().any()
            or raw_sizes.is_null().any()
            or (raw_sizes != raw_sizes.floor()).any()
        ):
            raise InputValidationError("lot sizes need unique assets and integer sizes")
    lotsizes = _account._lot_size_lookup(lot_sizes, config.default_buy_lot_size)
    prices_by_day = _account._price_lookup(prices)
    if "reference_price" in prices.columns:
        for row in prices.select("time", "asset_id", "reference_price").to_dicts():
            prices_by_day[row["time"]][row["asset_id"]]["reference_price"] = row[
                "reference_price"
            ]
    prefix_extra = {
        "weight_targets": weight_evidence,
        "decisions": decision_evidence,
        "lot_sizes": lot_sizes,
        "action_coverage": corporate_action_coverage,
    }

    def proof(through):
        base = _prefix_hash(
            plans,
            prices,
            execution_availability,
            actions,
            sessions_frame,
            config,
            through,
        )
        known_executions = [
            end for start, end in (schedule or {}).items() if start <= through
        ]
        frozen_calendar = sessions_frame.filter(pl.col("time") <= through)
        if known_executions:
            frozen_calendar = sessions_frame.filter(
                pl.col("time") <= max(through, *known_executions)
            )
        return _digest(
            {
                "base": base,
                "frozen_calendar": logical_content_hash(frozen_calendar),
                "frozen_schedule": {
                    start.isoformat(): end.isoformat()
                    for start, end in (schedule or {}).items()
                    if start <= through
                },
                "reference_price_mode": reference_price_mode,
                **{
                    name: logical_content_hash(causal_prefix_frame(frame, through))
                    for name, frame in prefix_extra.items()
                    if frame is not None
                },
            }
        )

    if checkpoint is not None:
        if checkpoint.initial_capital != initial_capital:
            raise InputValidationError("checkpoint initial capital differs")
        if checkpoint.prefix_hash != proof(checkpoint.time):
            raise InputValidationError("execution checkpoint causal prefix changed")
        actual_checkpoint = _account.complete_checkpoint_entitlements(
            checkpoint.actual,
            actions,
            historical_positions=historical_positions,
            historical_sessions=historical_sessions,
        )
        ideal_checkpoint = _account.complete_checkpoint_entitlements(
            checkpoint.ideal,
            actions,
            historical_positions=historical_ideal_positions,
            historical_sessions=historical_sessions,
        )
        actual = _account._restore_state(
            _account_config(config, initial_capital),
            initial_positions=None,
            initial_cash=None,
            checkpoint=actual_checkpoint,
        )
        ideal = _account._restore_state(
            _account_config(config, initial_capital, ideal=True),
            initial_positions=None,
            initial_cash=None,
            checkpoint=ideal_checkpoint,
        )
        actual_lots = checkpoint.actual_lots.to_dicts()
        ideal_lots = checkpoint.ideal_lots.to_dicts()
        pending = checkpoint.pending_transactions.to_dicts()
        entitlement_lots = checkpoint.entitlement_lots.to_dicts()
        corporate_pnl = checkpoint.corporate_pnl.to_dicts()
        for ledger, account_checkpoint, history in (
            ("actual", checkpoint.actual, historical_open_lots),
            ("ideal", checkpoint.ideal, historical_ideal_open_lots),
        ):
            known = (
                set(account_checkpoint.entitlements["action_id"].to_list())
                if ("action_id" in account_checkpoint.entitlements.columns)
                else set()
            )
            for action in actions.filter(
                (pl.col("record_date") <= checkpoint.time)
                & ~pl.col("action_id").is_in(sorted(known))
            ).to_dicts():
                if historical_sessions is not None and action["record_date"] not in (
                    historical_sessions["time"].to_list()
                ):
                    continue
                if history is None or "time" not in history.columns:
                    raise InputValidationError(
                        "new corporate attribution requires verified "
                        "record-date lot history"
                    )
                record_lots = history.filter(
                    pl.col("time") == action["record_date"]
                ).to_dicts()
                entitlement_lots.extend(_lot_entitlements(record_lots, action, ledger))
        processed = set(checkpoint.processed_plan_ids)
        totals = {
            name: getattr(checkpoint, name)
            for name in (
                "actual_realized_gross",
                "actual_realized_net",
                "ideal_realized_gross",
                "actual_income",
                "ideal_income",
                "fee_total",
            )
        }
        sessions = [day for day in sessions if day > checkpoint.time]
    else:
        actual = _account._restore_state(
            _account_config(config, initial_capital),
            initial_positions=None,
            initial_cash=None,
            checkpoint=None,
        )
        ideal = _account._restore_state(
            _account_config(config, initial_capital, ideal=True),
            initial_positions=None,
            initial_cash=None,
            checkpoint=None,
        )
        actual_lots, ideal_lots, pending, processed = [], [], [], set()
        entitlement_lots, corporate_pnl = [], []
        totals = dict.fromkeys(
            (
                "actual_realized_gross",
                "actual_realized_net",
                "ideal_realized_gross",
                "actual_income",
                "ideal_income",
                "fee_total",
            ),
            0.0,
        )
    if not sessions:
        raise InputValidationError("execution has no new market sessions")
    by_day = {}
    pending_ids = {row["plan_id"] for row in pending}
    if checkpoint is not None and weight_targets is None:
        supplied = {row["plan_id"]: row for row in plans.to_dicts()}
        for row in pending:
            if row.get("_retry"):
                continue
            original = supplied.get(row["plan_id"])
            if original is None or any(
                original[key] != row.get(key) for key in _PLAN_SCHEMA
            ):
                raise InputValidationError("frozen pending transaction changed")
    for row in plans.to_dicts():
        if row["plan_id"] not in processed and row["plan_id"] not in pending_ids:
            by_day.setdefault(row["execution_date"], []).append(row)
    # Frozen future decisions in the checkpoint are authoritative.
    for row in pending:
        by_day.setdefault(row["execution_date"], []).append(row)
    future = []
    emitted, orders, fills, ideal_fills, positions, values, returns = (
        _account._AccountRows() for _ in range(7)
    )
    matches, receivables, cashrows, attribution, weights = (
        _account._AccountRows() for _ in range(5)
    )
    actual_lot_history, ideal_lot_history, corporate_rows = (
        _account._AccountRows() for _ in range(3)
    )
    valuation_marks = _account._AccountRows()
    ex = _account._group_actions(actions, "ex_date")
    pay = _account._group_actions(actions, "cash_pay_date")
    listing = _account._group_actions(actions, "share_available_date")
    record = _account._group_actions(actions, "record_date")
    previous_actual = _account._state_equity(actual)
    previous_ideal = _account._state_equity(ideal)
    for completed, day in enumerate(sessions):
        check_canceled()
        index = all_sessions.index(day)
        day_prices = prices_by_day.get(day, {})
        for state, fifo, ledger in (
            (ideal, ideal_lots, "ideal"),
            (actual, actual_lots, "actual"),
        ):
            for action in ex.get(day, ()):
                allocations = _corporate_allocations(action, entitlement_lots, ledger)
                corporate_pnl.extend(allocations)
                for allocation in allocations:
                    corporate_rows.append(allocation)
            _account._release_settled_shares(state, day)
            action_rows = []
            _account._apply_corporate_actions(
                state,
                day,
                actions_by_ex=ex,
                actions_by_pay=pay,
                actions_by_list=listing,
                receivable_rows=action_rows,
            )
            _corporate_lots(
                fifo, action_rows, corporate_pnl, session=day, ledger=ledger
            )
            # Dividend income is recognized when the cash receivable arises.
            totals[f"{ledger}_income"] += sum(
                row["amount"] for row in action_rows if row["kind"] == "cash_created"
            )
            for row in action_rows:
                receivables.append({**row, "ledger": ledger})
            _account._mark_positions(state, day_prices, use="open")
        daily_fee = daily_slippage = 0.0
        todays = by_day.pop(day, [])
        seen = set()
        unique = []
        for row in todays:
            if row["plan_id"] not in seen:
                seen.add(row["plan_id"])
                unique.append(row)
        unique.sort(
            key=lambda row: (row["quantity"] > 0, row["asset_id"], row["plan_id"])
        )
        for row in unique:
            check_canceled()
            asset = row["asset_id"]
            requested = abs(row["quantity"])
            side = "buy" if row["quantity"] > 0 else "sell"
            price = day_prices.get(asset, {}).get("open")
            reference_price = row.get("reference_price")
            if reference_price is None:
                reference_price = day_prices.get(asset, {}).get("reference_price")
            if reference_price is None:
                reference_price = price
            is_retry = row.get("_retry", False)
            if not is_retry:
                if reference_price is None:
                    raise InputValidationError(
                        f"ideal plan {row['plan_id']} requires execution price"
                    )
                if side == "sell" and requested > ideal["positions"].get(asset, 0):
                    raise InputValidationError(
                        f"ideal plan {row['plan_id']} requires short inventory"
                    )
                if side == "buy" and requested * reference_price > ideal["cash"] + 1e-8:
                    raise InputValidationError(
                        f"ideal plan {row['plan_id']} is cash-infeasible"
                    )
                ideal_fill = _fill(
                    ideal,
                    row,
                    requested,
                    reference_price,
                    day,
                    index,
                    all_sessions,
                    config,
                    ideal_fills,
                    ideal=True,
                )
                gross, _ = _fifo_fill(ideal_lots, ideal_fill, matches, "ideal")
                totals["ideal_realized_gross"] += gross
                processed.add(row["plan_id"])
                emitted.append({key: row.get(key) for key in _PLAN_SCHEMA})
            can_buy, can_sell, reason = availability.get((day, asset), (True, True, ""))
            blocked = price is None or not (can_buy if side == "buy" else can_sell)
            qty = 0
            if blocked:
                reason = reason or "missing_execution_price"
            elif side == "sell":
                qty = min(requested, actual["available"].get(asset, 0))
                if qty < requested:
                    reason = "insufficient_available_inventory"
                if qty and qty * _quote(
                    config, side, qty, price, day, asset
                ).fill_price < (
                    _quote(config, side, qty, price, day, asset).fee - actual["cash"]
                ):
                    qty, reason = 0, "insufficient_cash_for_sell_cost"
            else:
                lot = lotsizes.get(asset, config.default_buy_lot_size)
                high, low = requested // lot, 0
                while low < high:
                    middle = (low + high + 1) // 2
                    quote = _quote(config, side, middle * lot, price, day, asset)
                    if (
                        middle * lot * quote.fill_price + quote.fee
                        <= actual["cash"] + 1e-8
                    ):
                        low = middle
                    else:
                        high = middle - 1
                qty = low * lot
                if qty < requested:
                    reason = "lot_or_cash_constraint"
            if qty:
                fill = _fill(
                    actual,
                    row,
                    qty,
                    price,
                    day,
                    index,
                    all_sessions,
                    config,
                    fills,
                    ideal=False,
                )
                gross, net = _fifo_fill(actual_lots, fill, matches, "actual")
                totals["actual_realized_gross"] += gross
                totals["actual_realized_net"] += net
                totals["fee_total"] += fill["explicit_cost"]
                daily_fee += fill["explicit_cost"]
                daily_slippage += fill["slippage_cost"]
            remainder = requested - qty
            retry = (
                config.retry_unfilled and remainder and index + 1 < len(all_sessions)
            )
            orders.append(
                {
                    "time": day,
                    "execution_date": day,
                    "plan_id": row["plan_id"],
                    "order_id": row["plan_id"],
                    "asset_id": asset,
                    "side": side,
                    "requested_quantity": requested,
                    "filled_quantity": qty,
                    "order_quantity": qty,
                    "unfilled_quantity": remainder,
                    "status": "filled"
                    if not remainder
                    else "pending"
                    if retry
                    else "expired",
                    "reason": reason or None,
                    "expires_at": None if retry or not remainder else day,
                }
            )
            if retry:
                retry_row = {
                    **row,
                    "execution_date": all_sessions[index + 1],
                    "quantity": remainder if side == "buy" else -remainder,
                    "_retry": True,
                }
                by_day.setdefault(retry_row["execution_date"], []).append(retry_row)
        for state, ledger in ((ideal, "ideal"), (actual, "actual")):
            _account._mark_positions(state, day_prices, use="close")
            _account._record_entitlements(day, state, record.get(day, ()))
            fifo = actual_lots if ledger == "actual" else ideal_lots
            for action in record.get(day, ()):
                entitlement_lots.extend(_lot_entitlements(fifo, action, ledger))
            marked_assets = {
                asset for asset, quantity in state["positions"].items() if quantity
            }
            marked_assets.update(row["asset_id"] for row in state["stock_receivables"])
            for asset in sorted(marked_assets):
                valuation_marks.append(
                    {
                        "time": day,
                        "ledger": ledger,
                        "asset_id": asset,
                        "mark_price": state["marks"].get(asset),
                    }
                )
            for asset, quantity in sorted(state["positions"].items()):
                if quantity:
                    positions.append(
                        {
                            "time": day,
                            "ledger": ledger,
                            "asset_id": asset,
                            "quantity": quantity,
                            "available_quantity": state["available"].get(asset, 0),
                            "mark_price": state["marks"].get(asset),
                            "close_price": state["marks"].get(asset),
                            "market_value": quantity * state["marks"].get(asset, 0),
                        }
                    )
        equity, target_equity = (
            _account._state_equity(actual),
            _account._state_equity(ideal),
        )
        actual_unrealized, ideal_unrealized = (
            _unrealized(actual_lots, actual),
            _unrealized(ideal_lots, ideal),
        )
        open_fees = sum(lot["entry_fee"] for lot in actual_lots)
        # Stock receivables have no purchase basis; their mark is an explicit
        # unrealized corporate-action contribution before share availability.
        actual_unrealized += _account._stock_receivable_value(actual)
        ideal_unrealized += _account._stock_receivable_value(ideal)
        actual_pnl = (
            totals["actual_realized_net"]
            + actual_unrealized
            - open_fees
            + totals["actual_income"]
        )
        ideal_pnl = (
            totals["ideal_realized_gross"] + ideal_unrealized + totals["ideal_income"]
        )
        if not math.isclose(
            equity - initial_capital, actual_pnl, abs_tol=1e-7, rel_tol=1e-9
        ):
            raise AssertionError("actual FIFO P&L does not reconcile account equity")
        if not math.isclose(
            target_equity - initial_capital, ideal_pnl, abs_tol=1e-7, rel_tol=1e-9
        ):
            raise AssertionError("ideal FIFO P&L does not reconcile account equity")
        net_return = equity / previous_actual - 1 if previous_actual > 0 else 0.0
        gross_return = target_equity / previous_ideal - 1 if previous_ideal > 0 else 0.0
        cost_return = daily_fee / previous_actual if previous_actual > 0 else 0.0
        returns.append(
            {
                "time": day,
                "gross_return": gross_return,
                "net_return": net_return,
                "net_pre_cost_return": net_return + cost_return,
                "cost_return": cost_return,
                "gross_nav": target_equity / initial_capital,
                "net_nav": equity / initial_capital,
            }
        )
        values.append(
            {
                "time": day,
                "cash": actual["cash"],
                "position_value": _account._position_value(actual),
                "equity": equity,
                "nav": equity / initial_capital,
                "units": initial_capital,
                "cash_receivable": _account._cash_receivable_value(actual),
                "stock_receivable_value": _account._stock_receivable_value(actual),
                "external_flow": 0.0,
                "pending_withdrawal": 0.0,
                "target_equity": target_equity,
                "target_nav": target_equity / initial_capital,
                "realized_gross_pnl": totals["actual_realized_gross"],
                "realized_net_pnl": totals["actual_realized_net"],
                "unrealized_gross_pnl": actual_unrealized,
                "unrealized_net_pnl": actual_unrealized - open_fees,
                "dividend_income": totals["actual_income"],
                "total_cost": totals["fee_total"],
                "reconciliation_error": equity - initial_capital - actual_pnl,
            }
        )
        cashrows.append({"time": day, "cash": actual["cash"]})
        attribution.append(
            {
                "time": day,
                "target_return": gross_return,
                "account_return": net_return,
                "cost_drag": -cost_return,
                "implementation_drag": net_return + cost_return - gross_return,
                "total_drag": net_return - gross_return,
                "slippage_cost": daily_slippage,
            }
        )
        for asset, quantity in sorted(actual["positions"].items()):
            if quantity:
                weight = quantity * actual["marks"][asset] / equity if equity else 0
                weights.append(
                    {
                        "time": day,
                        "asset_id": asset,
                        "weight": weight,
                        "actual_weight": weight,
                    }
                )
        if schedule and day in schedule:
            check_canceled()
            if day not in weight_targets:
                raise InputValidationError(
                    "rebalance decision has no complete saved weights"
                )
            generated = _account._build_stateful_decision_plan(
                decision_date=day,
                execution_date=schedule[day],
                prices=day_prices,
                state=actual,
                config=_account_config(config, initial_capital),
                lots=lotsizes,
                target_provider=lambda _context, snapshot=weight_targets[day]: snapshot,
            )
            for row in generated.to_dicts():
                delta = row["target_quantity"] - actual["positions"].get(
                    row["asset_id"], 0
                )
                if delta:
                    plan = {
                        "plan_id": _digest(
                            {
                                "decision": day,
                                "execution": row["execution_date"],
                                "asset": row["asset_id"],
                            }
                        ),
                        "decision_date": day,
                        "execution_date": row["execution_date"],
                        "asset_id": row["asset_id"],
                        "quantity": delta,
                        "reference_price": row["decision_price"]
                        if reference_price_mode == "decision"
                        else None,
                    }
                    by_day.setdefault(plan["execution_date"], []).append(plan)
        previous_actual, previous_ideal = equity, target_equity
        for fifo, history, state in (
            (actual_lots, actual_lot_history, actual),
            (ideal_lots, ideal_lot_history, ideal),
        ):
            for lot in fifo:
                if lot["quantity"]:
                    history.append(
                        {
                            "time": day,
                            **lot,
                            "mark_price": state["marks"][lot["asset_id"]],
                        }
                    )
        actual_lots = [lot for lot in actual_lots if lot["quantity"]]
        ideal_lots = [lot for lot in ideal_lots if lot["quantity"]]
        progress(completed + 1, len(sessions))
    for pending_day, rows in sorted(by_day.items()):
        if pending_day > sessions[-1]:
            future.extend(rows)
    final = sessions[-1]
    # Persist retry status separately from public transaction schema.
    future = [{**row, "_retry": bool(row.get("_retry"))} for row in future]
    pending_frame = (
        _frame(future, _PENDING_SCHEMA)
        .select(list(_PENDING_SCHEMA))
        .cast(_PENDING_SCHEMA)
    )
    result_checkpoint = ExecutionStateCheckpoint(
        time=final,
        actual=_account._checkpoint(final, actual),
        ideal=_account._checkpoint(final, ideal),
        actual_lots=_frame(
            [lot for lot in actual_lots if lot["quantity"]], _LOT_SCHEMA
        ),
        ideal_lots=_frame([lot for lot in ideal_lots if lot["quantity"]], _LOT_SCHEMA),
        pending_transactions=pending_frame,
        processed_plan_ids=tuple(sorted(processed)),
        initial_capital=initial_capital,
        prefix_hash=proof(final),
        entitlement_lots=_frame(entitlement_lots, _ENTITLEMENT_SCHEMA).cast(
            _ENTITLEMENT_SCHEMA
        ),
        corporate_pnl=_frame(corporate_pnl, _CORPORATE_PNL_SCHEMA).cast(
            _CORPORATE_PNL_SCHEMA
        ),
        **totals,
    )
    return_frame = _frame(returns)
    from .evaluation import return_statistics

    metrics = return_statistics(return_frame, annualization=config.annualization)
    metrics.update(
        {
            "total_cost": totals["fee_total"],
            "target_pnl": target_equity - initial_capital,
            "actual_pnl": equity - initial_capital,
            "implementation_shortfall": target_equity - equity,
        }
    )
    positions_frame = _frame(
        positions,
        {
            "time": pl.Date,
            "ledger": pl.String,
            "asset_id": pl.String,
            "quantity": pl.Int64,
            "available_quantity": pl.Int64,
            "mark_price": pl.Float64,
            "close_price": pl.Float64,
            "market_value": pl.Float64,
        },
    )
    tables = {
        "returns": return_frame,
        "account_value": _frame(values),
        "orders": _frame(orders, _ORDER_SCHEMA).cast(_ORDER_SCHEMA),
        "fills": _frame(fills, _FILL_SCHEMA).cast(_FILL_SCHEMA),
        "ideal_fills": _frame(ideal_fills, _FILL_SCHEMA).cast(_FILL_SCHEMA),
        "positions": positions_frame.filter(pl.col("ledger") == "actual").drop(
            "ledger"
        ),
        "ideal_positions": positions_frame.filter(pl.col("ledger") == "ideal").drop(
            "ledger"
        ),
        "cash": _frame(cashrows),
        "fifo_matches": _frame(matches, _MATCH_SCHEMA).cast(_MATCH_SCHEMA),
        "open_lots": result_checkpoint.actual_lots,
        "ideal_open_lots": result_checkpoint.ideal_lots,
        "open_lot_history": _frame(actual_lot_history, _LOT_HISTORY_SCHEMA).cast(
            _LOT_HISTORY_SCHEMA
        ),
        "ideal_open_lot_history": _frame(ideal_lot_history, _LOT_HISTORY_SCHEMA).cast(
            _LOT_HISTORY_SCHEMA
        ),
        "corporate_pnl": _frame(corporate_rows, _CORPORATE_PNL_SCHEMA).cast(
            _CORPORATE_PNL_SCHEMA
        ),
        "valuation_marks": _frame(
            valuation_marks,
            {
                "time": pl.Date,
                "ledger": pl.String,
                "asset_id": pl.String,
                "mark_price": pl.Float64,
            },
        ),
        "receivables": _frame(
            receivables,
            {
                "time": pl.Date,
                "action_id": pl.String,
                "asset_id": pl.String,
                "kind": pl.String,
                "amount": pl.Float64,
                "quantity": pl.Int64,
                "ledger": pl.String,
            },
        ),
        "attribution": _frame(attribution),
        "executable_weights": _frame(
            weights,
            {
                "time": pl.Date,
                "asset_id": pl.String,
                "weight": pl.Float64,
                "actual_weight": pl.Float64,
            },
        ),
        "transaction_plans": _frame(emitted, _PLAN_SCHEMA),
        "performance": return_frame.select(
            "time",
            pl.col("net_return").alias("account_return"),
            pl.col("net_return").alias("unit_return"),
            pl.col("net_nav").alias("performance_nav"),
        ),
        "external_flows": pl.DataFrame(schema={"time": pl.Date, "amount": pl.Float64}),
        "pending_withdrawals": pl.DataFrame(
            schema={"time": pl.Date, "amount": pl.Float64}
        ),
    }
    tables["transaction_pnl"] = _transaction_pnl(
        tables, actual, ideal, corporate=result_checkpoint.corporate_pnl, through=final
    )
    check_canceled()
    return ExecutionEvaluationResult(tables, metrics, result_checkpoint)


def _transaction_pnl(tables, actual, ideal, *, corporate=None, through=None):
    rows = {}
    for ledger, fills, lots, state in (
        ("target", tables["ideal_fills"], tables["ideal_open_lots"], ideal),
        ("actual", tables["fills"], tables["open_lots"], actual),
    ):
        for fill in fills.to_dicts():
            row = rows.setdefault(
                fill["plan_id"],
                {
                    "plan_id": fill["plan_id"],
                    "asset_id": fill["asset_id"],
                    "side": fill["side"],
                },
            )
            for name, value in (
                ("quantity", fill["quantity"]),
                ("notional", fill["notional"]),
                ("cost", fill["explicit_cost"]),
            ):
                key = f"{ledger}_{name}"
                row[key] = row.get(key, 0) + value
        for lot in lots.to_dicts():
            row = rows.setdefault(
                lot["plan_id"],
                {"plan_id": lot["plan_id"], "asset_id": lot["asset_id"], "side": "buy"},
            )
            key = f"{ledger}_unrealized_pnl"
            row[key] = row.get(key, 0) + (
                lot["quantity"] * (state["marks"][lot["asset_id"]] - lot["price"])
                - lot["entry_fee"]
            )
    for match in tables["fifo_matches"].to_dicts():
        ledger = "target" if match["ledger"] == "ideal" else "actual"
        row = rows.setdefault(
            match["sell_plan_id"],
            {
                "plan_id": match["sell_plan_id"],
                "asset_id": match["asset_id"],
                "side": "sell",
            },
        )
        key = f"{ledger}_realized_pnl"
        row[key] = row.get(key, 0) + match["net_pnl"]
    if corporate is not None:
        for event in corporate.to_dicts():
            ledger = "target" if event["ledger"] == "ideal" else "actual"
            row = rows.setdefault(
                event["plan_id"],
                {
                    "plan_id": event["plan_id"],
                    "asset_id": event["asset_id"],
                    "side": "buy",
                },
            )
            income_key = f"{ledger}_dividend_income"
            row[income_key] = row.get(income_key, 0) + event["cash_income"]
            if event["stock_quantity"] and event["share_available_date"] > through:
                state = ideal if ledger == "target" else actual
                key = f"{ledger}_unrealized_pnl"
                row[key] = row.get(key, 0) + event["stock_quantity"] * state[
                    "marks"
                ].get(event["asset_id"], 0)
    for row in rows.values():
        for ledger in ("target", "actual"):
            for name in (
                "quantity",
                "notional",
                "cost",
                "realized_pnl",
                "unrealized_pnl",
            ):
                row.setdefault(f"{ledger}_{name}", 0)
            row.setdefault(f"{ledger}_dividend_income", 0)
            pnl = (
                row.get(f"{ledger}_realized_pnl", 0)
                + row.get(f"{ledger}_unrealized_pnl", 0)
                + row[f"{ledger}_dividend_income"]
            )
            row[f"{ledger}_pnl"] = pnl
            notional = row.get(f"{ledger}_notional", 0)
            row[f"{ledger}_return"] = pnl / notional if notional else None
            row[f"{ledger}_return_reason"] = None if notional else "no_filled_notional"
    return (
        _frame([rows[key] for key in sorted(rows)], _PNL_SCHEMA)
        .select(list(_PNL_SCHEMA))
        .cast(_PNL_SCHEMA)
    )


def merge_execution_tables(
    previous: Mapping[str, pl.DataFrame],
    continuation: Mapping[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Append account history and recompute FIFO summaries without replaying.

    ``continuation`` contains sessions strictly after the preceding checkpoint.
    Open-lot tables are complete snapshots; transaction P&L is derived from the
    combined fills, matches and final marks. This is the single append path for
    execution result consumers, including Workbench and BTStore.
    """
    snapshots = {"open_lots", "ideal_open_lots", "transaction_pnl"}
    result = {}
    for name in sorted(set(previous) | set(continuation)):
        current = continuation.get(name)
        prior = previous.get(name)
        if name in snapshots:
            result[name] = current if current is not None else prior
        elif prior is None or prior.is_empty():
            result[name] = current if current is not None else prior
        elif current is None or current.is_empty():
            result[name] = prior
        else:
            if "time" in prior.columns and "time" in current.columns:
                if current["time"].min() <= prior["time"].max():
                    raise InputValidationError(
                        "execution continuation overlaps saved history"
                    )
            result[name] = pl.concat([prior, current], how="diagonal_relaxed")
    result["transaction_pnl"] = summarize_transaction_pnl(
        result, through=result["account_value"]["time"].max()
    )
    return result


def summarize_transaction_pnl(
    frames: Mapping[str, pl.DataFrame],
    *,
    through: date,
) -> pl.DataFrame:
    """Summarize saved transactions at a historical cutoff without replay.

    Only matured fill/match/corporate events are included. Lot snapshots use
    the last saved account session on/before ``through``, including its empty
    inventory, so a later sale/price/label can never enter an earlier view.
    """
    sessions = frames["account_value"].filter(pl.col("time") <= through)
    if sessions.is_empty():
        return pl.DataFrame(schema=_PNL_SCHEMA)
    cutoff = sessions["time"].max()
    selected = {
        "fills": frames["fills"].filter(pl.col("time") <= cutoff),
        "ideal_fills": frames["ideal_fills"].filter(pl.col("time") <= cutoff),
        "fifo_matches": frames["fifo_matches"].filter(pl.col("exit_date") <= cutoff),
        "open_lots": frames["open_lot_history"]
        .filter(pl.col("time") == cutoff)
        .select(list(_LOT_SCHEMA)),
        "ideal_open_lots": frames["ideal_open_lot_history"]
        .filter(pl.col("time") == cutoff)
        .select(list(_LOT_SCHEMA)),
    }
    corporate = frames["corporate_pnl"].filter(pl.col("time") <= cutoff)
    marks = frames["valuation_marks"].filter(pl.col("time") == cutoff)
    actual = {
        "marks": {
            row["asset_id"]: row["mark_price"]
            for row in marks.filter(pl.col("ledger") == "actual").to_dicts()
        }
    }
    ideal = {
        "marks": {
            row["asset_id"]: row["mark_price"]
            for row in marks.filter(pl.col("ledger") == "ideal").to_dicts()
        }
    }
    return _transaction_pnl(
        selected, actual, ideal, corporate=corporate, through=cutoff
    )


def execution_stress_scenarios(
    initial_capital: float,
    config: ExecutionConfig,
    *,
    session_lag: int,
) -> tuple[tuple[str, float, ExecutionConfig, int], ...]:
    """Return nine one-factor scenarios with explicit fixed account rules.

    Commission scenarios scale the proportional commission rate, preserving
    minimum fees and tax. Slippage scenarios scale the caller's supplied rates;
    zero stays zero. Opaque custom quote rules require caller-declared scenarios
    because their commission component cannot be inferred from a total fee.
    """
    if not math.isfinite(initial_capital) or initial_capital <= 0 or session_lag < 1:
        raise InputValidationError("stress requires positive capital and execution lag")
    if config.cost_rule is not None:
        raise InputValidationError(
            "custom cost rules require explicit caller-declared stress scenarios"
        )
    scenarios = []
    for multiple in (2, 5, 10):
        scenarios.append(
            (
                f"capital_{multiple}",
                initial_capital * multiple,
                replace(
                    config,
                    fixed_notional=(
                        config.fixed_notional * multiple
                        if config.fixed_notional is not None
                        else None
                    ),
                ),
                session_lag,
            )
        )
    for multiple in (2, 4):
        scenarios.append(
            (
                f"commission_{multiple}",
                initial_capital,
                replace(config, rate=config.rate * multiple),
                session_lag,
            )
        )
        scenarios.append(
            (
                f"slippage_{multiple}",
                initial_capital,
                replace(
                    config,
                    buy_slippage_rate=config.buy_slippage_rate * multiple,
                    sell_slippage_rate=config.sell_slippage_rate * multiple,
                ),
                session_lag,
            )
        )
    for delay in (1, 4):
        scenarios.append(
            (f"delay_{delay}", initial_capital, config, session_lag + delay)
        )
    return tuple(scenarios)


def _json_value(value):
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def logical_content_hash(frame: pl.DataFrame) -> str:
    """Hash canonical logical rows and schema, independent of Arrow chunks."""
    columns = sorted(frame.columns)
    sorted_frame = frame.select(columns).sort(columns) if columns else frame
    digest = hashlib.sha256()
    digest.update(
        _digest(
            [(name, str(dtype)) for name, dtype in sorted_frame.schema.items()]
        ).encode()
    )
    for row in sorted_frame.iter_rows(named=True):
        digest.update(_digest(row).encode())
    return digest.hexdigest()


def causal_prefix_frame(frame, through, *, assets=None):
    """Canonical historical rows without future effective/availability facts."""
    if frame is None:
        return pl.DataFrame()
    if assets is not None and "asset_id" in frame.columns:
        values = assets.to_list() if isinstance(assets, pl.Series) else list(assets)
        frame = frame.filter(pl.col("asset_id").is_in(values))
    column = next(
        (
            name
            for name in ("time", "ex_date", "rebalance_date", "execution_date")
            if name in frame.columns
        ),
        None,
    )
    if column:
        if column == "execution_date" and "decision_date" in frame.columns:
            frame = frame.filter(
                pl.coalesce("decision_date", "execution_date") <= through
            )
        else:
            frame = frame.filter(pl.col(column) <= through)
    for name in ("available_date", "version_available_date"):
        if name in frame.columns:
            frame = frame.filter(pl.col(name).is_null() | (pl.col(name) <= through))
    return frame.sort(frame.columns) if frame.columns else frame


def input_prefix(*, through, settings, assets=None, **inputs):
    """Prove supplied neutral account inputs; never resolve a backend source."""
    evidence = {}
    for name, value in sorted(inputs.items()):
        if isinstance(value, pl.DataFrame):
            if name == "calendar" and "time" in value.columns:
                calendar_prefix = value.filter(pl.col("time") <= through)
                lag = int(settings.get("session_lag", 0))
                if lag:
                    calendar_prefix = pl.concat(
                        [
                            calendar_prefix,
                            value.filter(pl.col("time") > through)
                            .sort("time")
                            .head(lag),
                        ]
                    )
                evidence[name] = logical_content_hash(calendar_prefix)
                continue
            evidence[name] = logical_content_hash(
                causal_prefix_frame(value, through, assets=assets)
            )
        elif isinstance(value, Mapping):
            evidence[name] = {
                key: item
                for key, item in value.items()
                if key not in {"days", "decision_days"}
            }
            for key in ("days", "decision_days"):
                if key in value:
                    evidence[name][key] = {
                        day: digest
                        for day, digest in value[key].items()
                        if day <= through.isoformat()
                    }
    return _digest(
        {
            "schema": "execution.causal_prefix.v1",
            "through": through,
            "settings": settings,
            "inputs": evidence,
        }
    )


def save_checkpoint(checkpoint):
    """Encode public account/execution checkpoints as tables plus JSON metadata."""
    if isinstance(checkpoint, AccountStateCheckpoint):
        frames, metadata = {}, {"checkpoint_type": "account"}
        for item in fields(checkpoint):
            value = getattr(checkpoint, item.name)
            if isinstance(value, pl.DataFrame):
                frames[f"checkpoint_{item.name}"] = value
            else:
                metadata[item.name] = _json_value(value)
        return frames, metadata
    if not isinstance(checkpoint, ExecutionStateCheckpoint):
        raise InputValidationError("unsupported checkpoint type")
    frames, metadata = {}, {"checkpoint_type": "execution"}
    for item in fields(checkpoint):
        value = getattr(checkpoint, item.name)
        if isinstance(value, AccountStateCheckpoint):
            nested_frames, nested_metadata = save_checkpoint(value)
            frames.update(
                {f"{item.name}_{key}": frame for key, frame in nested_frames.items()}
            )
            metadata[item.name] = nested_metadata
        elif isinstance(value, pl.DataFrame):
            frames[f"checkpoint_{item.name}"] = value
        else:
            metadata[item.name] = _json_value(value)
    return frames, metadata


def load_checkpoint(frames, metadata):
    """Decode only declared public checkpoint fields; reject malformed records."""
    data = copy.deepcopy(dict(metadata))
    kind = data.pop("checkpoint_type", "account")
    data["time"] = date.fromisoformat(data["time"])
    if kind == "account":
        if data.get("target_revision_time"):
            data["target_revision_time"] = date.fromisoformat(
                data["target_revision_time"]
            )
        names = {item.name for item in fields(AccountStateCheckpoint)}
        data.update(
            {
                name.removeprefix("checkpoint_"): frame
                for name, frame in frames.items()
                if name.startswith("checkpoint_")
                and name.removeprefix("checkpoint_") in names
            }
        )
        return AccountStateCheckpoint(**data)
    if kind != "execution":
        raise InputValidationError("unknown checkpoint type")
    for ledger in ("actual", "ideal"):
        data[ledger] = load_checkpoint(
            {
                name.removeprefix(f"{ledger}_"): frame
                for name, frame in frames.items()
                if name.startswith(f"{ledger}_")
            },
            data[ledger],
        )
    for name in (
        "actual_lots",
        "ideal_lots",
        "pending_transactions",
        "entitlement_lots",
        "corporate_pnl",
    ):
        data[name] = frames[f"checkpoint_{name}"]
    data["processed_plan_ids"] = tuple(data["processed_plan_ids"])
    return ExecutionStateCheckpoint(**data)
