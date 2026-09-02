"""Deterministic whole-share account simulation on unadjusted market prices."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import polars as pl

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


@dataclass(frozen=True, slots=True)
class AccountBacktestResult:
    """Auditable quantities, ledgers, orders, fills, NAV, and implementation drag."""

    target_weights: pl.DataFrame
    target_positions: pl.DataFrame
    orders: pl.DataFrame
    fills: pl.DataFrame
    positions: pl.DataFrame
    cash: pl.DataFrame
    receivables: pl.DataFrame
    external_flows: pl.DataFrame
    pending_withdrawals: pl.DataFrame
    account_value: pl.DataFrame
    performance: pl.DataFrame
    executable_weights: pl.DataFrame
    attribution: pl.DataFrame
    final_checkpoint: AccountStateCheckpoint


def run_account_backtest(
    target_weights: pl.DataFrame,
    market_prices: pl.DataFrame,
    *,
    corporate_action_coverage: pl.DataFrame,
    config: AccountBacktestConfig | None = None,
    execution_availability: pl.DataFrame | None = None,
    lot_sizes: pl.DataFrame | None = None,
    corporate_actions: pl.DataFrame | None = None,
    initial_positions: pl.DataFrame | None = None,
    initial_cash: float | None = None,
    checkpoint: AccountStateCheckpoint | None = None,
) -> AccountBacktestResult:
    """Run a daily deterministic whole-share account simulation.

    Targets are execution-date weights. Prices must be unadjusted open/close
    observations. Corporate-action coverage is mandatory so missing event data
    can never be silently replaced by adjusted prices.
    """

    return _run_account_backtest(
        target_weights,
        market_prices,
        corporate_action_coverage=corporate_action_coverage,
        config=config,
        execution_availability=execution_availability,
        lot_sizes=lot_sizes,
        corporate_actions=corporate_actions,
        initial_positions=initial_positions,
        initial_cash=initial_cash,
        checkpoint=checkpoint,
        target_position_plans=None,
    )


def run_planned_account_backtest(
    target_position_plans: pl.DataFrame,
    market_prices: pl.DataFrame,
    *,
    corporate_action_coverage: pl.DataFrame,
    config: AccountBacktestConfig | None = None,
    execution_availability: pl.DataFrame | None = None,
    lot_sizes: pl.DataFrame | None = None,
    corporate_actions: pl.DataFrame | None = None,
    initial_positions: pl.DataFrame | None = None,
    initial_cash: float | None = None,
    checkpoint: AccountStateCheckpoint | None = None,
) -> AccountBacktestResult:
    """Execute immutable decision-close target quantities at the next open.

    Each plan row carries ``decision_date``, ``execution_date``, ``asset_id``,
    ``target_weight``, ``sizing_notional``, ``decision_price``, and
    ``target_quantity``.  The execution session may constrain or reduce an
    order, but it never converts the target weight into a new quantity using
    the execution open.  Any unfilled remainder expires with that session.
    """

    plans = _validate_target_position_plans(target_position_plans)
    targets = plans.select(
        pl.col("execution_date").alias(TIME),
        ASSET_ID,
        pl.col("target_weight").alias("weight"),
    )
    return _run_account_backtest(
        targets,
        market_prices,
        corporate_action_coverage=corporate_action_coverage,
        config=config,
        execution_availability=execution_availability,
        lot_sizes=lot_sizes,
        corporate_actions=corporate_actions,
        initial_positions=initial_positions,
        initial_cash=initial_cash,
        checkpoint=checkpoint,
        target_position_plans=plans,
    )


def _run_account_backtest(
    target_weights: pl.DataFrame,
    market_prices: pl.DataFrame,
    *,
    corporate_action_coverage: pl.DataFrame,
    config: AccountBacktestConfig | None,
    execution_availability: pl.DataFrame | None,
    lot_sizes: pl.DataFrame | None,
    corporate_actions: pl.DataFrame | None,
    initial_positions: pl.DataFrame | None,
    initial_cash: float | None,
    checkpoint: AccountStateCheckpoint | None,
    target_position_plans: pl.DataFrame | None,
) -> AccountBacktestResult:
    resolved_config = config or AccountBacktestConfig()
    prices = _validate_market_prices(market_prices)
    targets = _validate_target_weights(target_weights)
    calendar = prices.get_column(TIME).unique().sort().to_list()
    if checkpoint is not None:
        calendar = [value for value in calendar if value > checkpoint.time]
    if not calendar:
        raise InputValidationError("account backtest has no market sessions to run")
    _validate_corporate_action_coverage(corporate_action_coverage, calendar)
    availability = _availability_lookup(execution_availability)
    lots = _lot_size_lookup(lot_sizes, resolved_config.default_buy_lot_size)
    actions = _validate_corporate_actions(corporate_actions)
    actions_by_record = _group_actions(actions, "record_date")
    actions_by_ex = _group_actions(actions, "ex_date")
    actions_by_pay = _group_actions(actions, "cash_pay_date")
    actions_by_list = _group_actions(actions, "share_available_date")
    prices_by_date = _price_lookup(prices)
    targets_by_date = _target_lookup(targets)
    plans_by_date = _target_position_plan_lookup(target_position_plans)

    state = _restore_state(
        resolved_config,
        initial_positions=initial_positions,
        initial_cash=initial_cash,
        checkpoint=checkpoint,
    )
    rows: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "target_weights",
            "target_positions",
            "orders",
            "fills",
            "positions",
            "cash",
            "receivables",
            "external_flows",
            "pending_withdrawals",
            "account_value",
            "performance",
            "executable_weights",
            "attribution",
        )
    }
    previous_equity = _state_equity(state)
    previous_nav = previous_equity / state["units"]

    for session_index, session in enumerate(calendar):
        session_prices = prices_by_date[session]
        _release_settled_shares(state, session)
        _apply_corporate_actions(
            state,
            session,
            actions_by_ex=actions_by_ex,
            actions_by_pay=actions_by_pay,
            actions_by_list=actions_by_list,
            receivable_rows=rows["receivables"],
        )
        _mark_positions(state, session_prices, use="open")
        preflow_equity = _state_equity(state)
        external_flow = 0.0
        is_target_session = session in targets_by_date
        if resolved_config.capital_mode == "fixed_notional" and is_target_session:
            external_flow += _rebalance_external_capital(
                state,
                resolved_config.fixed_notional,
                session,
                rows["external_flows"],
            )

        if is_target_session:
            state["latest_target"] = targets_by_date[session]
            state["target_revision_time"] = (
                plans_by_date[session]["decision_date"]
                if session in plans_by_date
                else session
            )
        latest_target: dict[str, float] = state["latest_target"]
        sizing_capital = (
            resolved_config.fixed_notional
            if resolved_config.capital_mode == "fixed_notional"
            else _state_equity(state)
        )
        if is_target_session:
            if session in plans_by_date:
                state["pending_target_positions"] = _planned_positions(
                    session,
                    plans_by_date[session],
                    session_prices,
                    rows["target_weights"],
                    rows["target_positions"],
                )
            else:
                state["pending_target_positions"] = _desired_positions(
                    session,
                    sizing_capital,
                    latest_target,
                    state,
                    session_prices,
                    lots,
                    rows["target_weights"],
                    rows["target_positions"],
                )
        desired = state["pending_target_positions"]
        daily_cost = 0.0
        withdrawal_flow = 0.0
        if is_target_session or desired:
            daily_cost, withdrawal_flow, pending = _execute_rebalance(
                session,
                session_index,
                calendar,
                desired,
                state,
                session_prices,
                availability,
                lots,
                resolved_config,
                rows["orders"],
                rows["fills"],
                rows["external_flows"],
                expire_unfilled=session in plans_by_date,
            )
            state["pending_target_positions"] = pending
        external_flow += withdrawal_flow

        _mark_positions(state, session_prices, use="close")
        equity = _state_equity(state)
        if equity < -1e-8 or state["cash"] < -1e-8:
            raise AssertionError("account engine produced negative cash or equity")
        nav = equity / state["units"] if state["units"] > 0 else 0.0
        daily_return = (
            (equity - previous_equity - external_flow) / previous_equity
            if previous_equity > 0
            else 0.0
        )
        unit_return = nav / previous_nav - 1.0 if previous_nav > 0 else 0.0
        target_return = _target_open_to_close_return(latest_target, session_prices)
        cost_drag = -daily_cost / max(preflow_equity, 1e-12)
        rows["attribution"].append(
            {
                TIME: session,
                "target_return": target_return,
                "account_return": daily_return,
                "implementation_drag": daily_return - cost_drag - target_return,
                "cost_drag": cost_drag,
                "total_drag": daily_return - target_return,
            }
        )
        rows["account_value"].append(
            {
                TIME: session,
                "cash": state["cash"],
                "position_value": _position_value(state),
                "cash_receivable": _cash_receivable_value(state),
                "stock_receivable_value": _stock_receivable_value(state),
                "equity": equity,
                "units": state["units"],
                "nav": nav,
                "external_flow": external_flow,
                "pending_withdrawal": state["pending_withdrawal"],
            }
        )
        rows["performance"].append(
            {
                TIME: session,
                "account_return": daily_return,
                "unit_return": unit_return,
                "performance_nav": nav / resolved_config.nav_base,
            }
        )
        _record_end_of_day_state(session, state, equity, rows)
        _record_entitlements(
            session,
            state,
            actions_by_record.get(session, ()),
        )
        previous_equity = equity
        previous_nav = nav

    final_time = calendar[-1]
    checkpoint_result = _checkpoint(final_time, state)
    return AccountBacktestResult(
        target_weights=_rows_frame(rows["target_weights"]),
        target_positions=_rows_frame(rows["target_positions"]),
        orders=_rows_frame(rows["orders"]),
        fills=_rows_frame(rows["fills"]),
        positions=_rows_frame(rows["positions"]),
        cash=_rows_frame(rows["cash"]),
        receivables=_rows_frame(rows["receivables"]),
        external_flows=_rows_frame(rows["external_flows"]),
        pending_withdrawals=_rows_frame(rows["pending_withdrawals"]),
        account_value=_rows_frame(rows["account_value"]),
        performance=_rows_frame(rows["performance"]),
        executable_weights=_rows_frame(rows["executable_weights"]),
        attribution=_rows_frame(rows["attribution"]),
        final_checkpoint=checkpoint_result,
    )


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


def _validate_target_position_plans(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "decision_date",
        "execution_date",
        ASSET_ID,
        "target_weight",
        "sizing_notional",
        "decision_price",
        "target_quantity",
    }
    if not isinstance(frame, pl.DataFrame) or not required.issubset(frame.columns):
        raise InputValidationError(
            "target_position_plans requires decision_date, execution_date, "
            "asset_id, target_weight, sizing_notional, decision_price, and "
            "target_quantity columns"
        )
    result = frame.select(*sorted(required)).with_columns(
        pl.col("decision_date").cast(pl.Date, strict=False),
        pl.col("execution_date").cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("target_weight").cast(pl.Float64, strict=False),
        pl.col("sizing_notional").cast(pl.Float64, strict=False),
        pl.col("decision_price").cast(pl.Float64, strict=False),
        pl.col("target_quantity").cast(pl.Int64, strict=False),
    )
    if result.select(
        pl.struct("execution_date", ASSET_ID).is_duplicated().any()
    ).item():
        raise InputValidationError(
            "target_position_plans must be unique by (execution_date, asset_id)"
        )
    invalid = result.filter(
        pl.col("decision_date").is_null()
        | pl.col("execution_date").is_null()
        | (pl.col("decision_date") >= pl.col("execution_date"))
        | pl.col(ASSET_ID).is_null()
        | (pl.col(ASSET_ID).str.len_chars() == 0)
        | pl.col("target_weight").is_null()
        | ~pl.col("target_weight").is_finite()
        | (pl.col("target_weight") < 0)
        | pl.col("sizing_notional").is_null()
        | ~pl.col("sizing_notional").is_finite()
        | (pl.col("sizing_notional") <= 0)
        | pl.col("decision_price").is_null()
        | ~pl.col("decision_price").is_finite()
        | (pl.col("decision_price") <= 0)
        | pl.col("target_quantity").is_null()
        | (pl.col("target_quantity") < 0)
    )
    if invalid.height:
        raise InputValidationError("target_position_plans contains invalid values")
    inconsistent = (
        result.group_by("execution_date")
        .agg(
            pl.col("decision_date").n_unique().alias("decision_dates"),
            pl.col("sizing_notional").n_unique().alias("notionals"),
            pl.col("target_weight").sum().alias("total_weight"),
        )
        .filter(
            (pl.col("decision_dates") != 1)
            | (pl.col("notionals") != 1)
            | (pl.col("total_weight") > 1.0 + 1e-8)
        )
    )
    if inconsistent.height:
        raise InputValidationError(
            "each execution_date requires one decision_date, one sizing_notional, "
            "and target weights summing to at most one"
        )
    return result.sort(["execution_date", ASSET_ID])


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
            "target_revision_time": checkpoint.time,
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


def _execute_rebalance(
    session: date,
    session_index: int,
    calendar: list[date],
    desired: dict[str, int | None],
    state: dict[str, Any],
    prices: dict[str, dict[str, float | None]],
    availability: dict[tuple[date, str], tuple[bool, bool, str]],
    lots: dict[str, int],
    config: AccountBacktestConfig,
    order_rows: list[dict[str, Any]],
    fill_rows: list[dict[str, Any]],
    flow_rows: list[dict[str, Any]],
    *,
    expire_unfilled: bool = False,
) -> tuple[float, float, dict[str, int | None]]:
    total_cost = 0.0
    pending: dict[str, int | None] = {}
    assets = sorted(set(state["positions"]) | set(desired))
    for asset_id in assets:
        current = state["positions"].get(asset_id, 0)
        target = desired.get(asset_id)
        if target is None or current <= target:
            continue
        requested = current - target
        sellable = state["available"].get(asset_id, 0)
        lot = lots.get(asset_id, config.default_buy_lot_size)
        executable = min(requested, sellable)
        settlement_blocked = sellable < requested
        if target != 0 and executable < current:
            executable = executable // lot * lot
        can_sell = availability.get((session, asset_id), (True, True, ""))[1]
        reason = availability.get((session, asset_id), (True, True, ""))[2]
        open_price = prices.get(asset_id, {}).get("open")
        market_blocked = not can_sell or open_price is None
        if market_blocked:
            executable = 0
            reason = reason or "missing_open_price"
        elif settlement_blocked:
            reason = reason or "t_plus_one_unavailable"
        elif executable < requested:
            reason = "cash_or_lot_constraint"
        order = _order_row(
            session,
            state,
            asset_id,
            "sell",
            requested,
            executable,
            reason,
            expire_unfilled=expire_unfilled,
        )
        order_rows.append(order)
        if executable:
            fill_cost = _apply_fill(
                session,
                asset_id,
                "sell",
                executable,
                float(open_price),
                state,
                config,
                calendar,
                session_index,
                fill_rows,
                order["order_id"],
            )
            total_cost += fill_cost
        if (
            requested > executable
            and config.retry_blocked_orders
            and not expire_unfilled
            and (market_blocked or settlement_blocked)
        ):
            pending[asset_id] = target

    withdrawal_flow = 0.0
    if config.capital_mode == "fixed_notional":
        withdrawal_flow = _pay_pending_withdrawal(state, session, flow_rows)

    buy_plans: dict[str, int] = {}
    while True:
        candidates: list[tuple[float, str, float]] = []
        for asset_id, target in desired.items():
            if target is None:
                continue
            current = state["positions"].get(asset_id, 0) + buy_plans.get(asset_id, 0)
            lot = lots.get(asset_id, config.default_buy_lot_size)
            if current + lot > target:
                continue
            can_buy, _, _ = availability.get((session, asset_id), (True, True, ""))
            open_price = prices.get(asset_id, {}).get("open")
            if not can_buy or open_price is None:
                continue
            old_quantity = buy_plans.get(asset_id, 0)
            new_quantity = old_quantity + lot
            incremental_cash = _buy_cash_cost(
                new_quantity, float(open_price), config
            ) - _buy_cash_cost(old_quantity, float(open_price), config)
            if incremental_cash > state["cash"] + 1e-9:
                continue
            unit_value = lot * float(open_price)
            gap_value = (target - current) * float(open_price)
            improvement = gap_value**2 - (gap_value - unit_value) ** 2
            if improvement > 0:
                candidates.append((improvement, asset_id, incremental_cash))
        if not candidates:
            break
        _, selected, incremental_cash = max(
            candidates, key=lambda value: (value[0], _reverse_asset_id(value[1]))
        )
        buy_plans[selected] = buy_plans.get(selected, 0) + lots.get(
            selected, config.default_buy_lot_size
        )
        state["cash"] -= incremental_cash

    # The planning loop reserved exact cash. Restore it before posting real fills.
    for asset_id, quantity in buy_plans.items():
        open_price = float(prices[asset_id]["open"])
        state["cash"] += _buy_cash_cost(quantity, open_price, config)
    for asset_id in sorted(set(desired) | set(buy_plans)):
        target = desired.get(asset_id)
        current = state["positions"].get(asset_id, 0)
        requested = max(0, (target or current) - current) if target is not None else 0
        planned = buy_plans.get(asset_id, 0)
        can_buy, _, reason = availability.get((session, asset_id), (True, True, ""))
        if prices.get(asset_id, {}).get("open") is None:
            reason = "missing_open_price"
        elif not can_buy:
            reason = reason or "buy_blocked"
        elif planned < requested:
            reason = "cash_or_lot_constraint"
        order = _order_row(
            session,
            state,
            asset_id,
            "buy",
            requested,
            planned,
            reason,
            expire_unfilled=expire_unfilled,
        )
        if requested or planned:
            order_rows.append(order)
        if planned:
            total_cost += _apply_fill(
                session,
                asset_id,
                "buy",
                planned,
                float(prices[asset_id]["open"]),
                state,
                config,
                calendar,
                session_index,
                fill_rows,
                order["order_id"],
            )
        if (
            requested > planned
            and config.retry_blocked_orders
            and not expire_unfilled
            and (not can_buy or prices.get(asset_id, {}).get("open") is None)
        ):
            pending[asset_id] = target
    return total_cost, withdrawal_flow, pending


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
    fill_rows: list[dict[str, Any]],
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


def _desired_positions(
    session: date,
    sizing_capital: float,
    target: dict[str, float],
    state: dict[str, Any],
    prices: dict[str, dict[str, float | None]],
    lots: dict[str, int],
    target_weight_rows: list[dict[str, Any]],
    target_position_rows: list[dict[str, Any]],
) -> dict[str, int | None]:
    desired: dict[str, int | None] = {}
    for asset_id in sorted(set(target) | set(state["positions"])):
        weight = target.get(asset_id, 0.0)
        target_weight_rows.append(
            {
                TIME: session,
                "target_revision_time": state["target_revision_time"],
                ASSET_ID: asset_id,
                "weight": weight,
            }
        )
        open_price = prices.get(asset_id, {}).get("open")
        if open_price is None:
            desired[asset_id] = None
            quantity = None
        else:
            lot = lots.get(asset_id, 1)
            quantity = int((weight * sizing_capital / float(open_price)) // lot * lot)
            desired[asset_id] = quantity
        target_position_rows.append(
            {
                TIME: session,
                "target_revision_time": state["target_revision_time"],
                ASSET_ID: asset_id,
                "target_weight": weight,
                "sizing_capital": sizing_capital,
                "open_price": open_price,
                "target_quantity": quantity,
            }
        )
    return desired


def _planned_positions(
    session: date,
    plan: dict[str, Any],
    prices: dict[str, dict[str, float | None]],
    target_weight_rows: list[dict[str, Any]],
    target_position_rows: list[dict[str, Any]],
) -> dict[str, int]:
    decision_date = plan["decision_date"]
    sizing_notional = float(plan["sizing_notional"])
    desired: dict[str, int] = {}
    for asset_id, row in sorted(plan["positions"].items()):
        weight = float(row["target_weight"])
        quantity = int(row["target_quantity"])
        desired[asset_id] = quantity
        target_weight_rows.append(
            {
                TIME: session,
                "target_revision_time": decision_date,
                ASSET_ID: asset_id,
                "weight": weight,
            }
        )
        target_position_rows.append(
            {
                TIME: session,
                "target_revision_time": decision_date,
                "decision_date": decision_date,
                "execution_date": session,
                ASSET_ID: asset_id,
                "target_weight": weight,
                "sizing_capital": sizing_notional,
                "decision_price": float(row["decision_price"]),
                "open_price": prices.get(asset_id, {}).get("open"),
                "target_quantity": quantity,
            }
        )
    return desired


def _apply_corporate_actions(
    state: dict[str, Any],
    session: date,
    *,
    actions_by_ex: dict[date, tuple[dict[str, Any], ...]],
    actions_by_pay: dict[date, tuple[dict[str, Any], ...]],
    actions_by_list: dict[date, tuple[dict[str, Any], ...]],
    receivable_rows: list[dict[str, Any]],
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


def _rebalance_external_capital(
    state: dict[str, Any],
    fixed_notional: float,
    session: date,
    flow_rows: list[dict[str, Any]],
) -> float:
    equity = _state_equity(state)
    if equity < fixed_notional - 1e-8:
        state["pending_withdrawal"] = 0.0
        return _post_external_flow(
            state,
            session,
            fixed_notional - equity,
            "fixed_notional_injection",
            flow_rows,
        )
    state["pending_withdrawal"] = max(equity - fixed_notional, 0.0)
    return _pay_pending_withdrawal(state, session, flow_rows)


def _pay_pending_withdrawal(
    state: dict[str, Any],
    session: date,
    flow_rows: list[dict[str, Any]],
) -> float:
    amount = min(state["cash"], state["pending_withdrawal"])
    if amount <= 1e-12:
        return 0.0
    state["pending_withdrawal"] -= amount
    return _post_external_flow(
        state, session, -amount, "fixed_notional_withdrawal", flow_rows
    )


def _post_external_flow(
    state: dict[str, Any],
    session: date,
    amount: float,
    reason: str,
    rows: list[dict[str, Any]],
) -> float:
    equity = _state_equity(state)
    nav = equity / state["units"]
    units_delta = amount / nav
    state["cash"] += amount
    state["units"] += units_delta
    rows.append(
        {TIME: session, "amount": amount, "units_delta": units_delta, "reason": reason}
    )
    return amount


def _record_end_of_day_state(
    session: date,
    state: dict[str, Any],
    equity: float,
    rows: dict[str, list[dict[str, Any]]],
) -> None:
    for asset_id in sorted(state["positions"]):
        quantity = state["positions"][asset_id]
        if quantity == 0:
            continue
        mark = state["marks"].get(asset_id)
        value = quantity * mark if mark is not None else 0.0
        rows["positions"].append(
            {
                TIME: session,
                ASSET_ID: asset_id,
                "quantity": quantity,
                "available_quantity": state["available"].get(asset_id, 0),
                "mark_price": mark,
                "market_value": value,
            }
        )
        rows["executable_weights"].append(
            {
                TIME: session,
                ASSET_ID: asset_id,
                "weight": value / equity if equity else 0.0,
                "actual_weight": value / equity if equity else 0.0,
            }
        )
    rows["cash"].append(
        {
            TIME: session,
            "available_cash": state["cash"],
            "cash_receivable": _cash_receivable_value(state),
        }
    )
    if state["pending_withdrawal"] > 1e-12:
        rows["pending_withdrawals"].append(
            {TIME: session, "amount": state["pending_withdrawal"], "status": "pending"}
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
        if quantity
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


def _buy_cash_cost(
    quantity: int, open_price: float, config: AccountBacktestConfig
) -> float:
    if quantity == 0:
        return 0.0
    fill_price = open_price * (1 + config.transaction_cost.slippage_for("buy"))
    notional = quantity * fill_price
    return (
        notional
        + max(config.transaction_cost.min_fee, notional * config.transaction_cost.rate)
        + notional * config.transaction_cost.transfer_fee_rate
    )


def _target_open_to_close_return(
    target: dict[str, float],
    prices: dict[str, dict[str, float | None]],
) -> float:
    return sum(
        weight * (float(item["close"]) / float(item["open"]) - 1.0)
        for asset_id, weight in target.items()
        if (item := prices.get(asset_id, {})).get("open") is not None
        and item.get("close") is not None
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


def _target_lookup(frame: pl.DataFrame) -> dict[date, dict[str, float]]:
    result: dict[date, dict[str, float]] = {}
    for row in frame.iter_rows(named=True):
        result.setdefault(row[TIME], {})[row[ASSET_ID]] = row["weight"]
    return result


def _target_position_plan_lookup(
    frame: pl.DataFrame | None,
) -> dict[date, dict[str, Any]]:
    result: dict[date, dict[str, Any]] = {}
    if frame is None:
        return result
    for row in frame.iter_rows(named=True):
        execution_date = row["execution_date"]
        plan = result.setdefault(
            execution_date,
            {
                "decision_date": row["decision_date"],
                "sizing_notional": row["sizing_notional"],
                "positions": {},
            },
        )
        plan["positions"][row[ASSET_ID]] = row
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


def _order_row(
    session: date,
    state: dict[str, Any],
    asset_id: str,
    side: str,
    requested: int,
    executable: int,
    reason: str,
    *,
    expire_unfilled: bool = False,
) -> dict[str, Any]:
    revision = state["target_revision_time"]
    seed = f"{session}|{revision}|{asset_id}|{side}".encode()
    order_id = hashlib.sha256(seed).hexdigest()[:24]
    status = (
        "filled"
        if requested == executable and executable
        else "reduced"
        if executable
        else "expired"
        if expire_unfilled
        else "pending"
    )
    return {
        TIME: session,
        "target_revision_time": revision,
        "order_id": order_id,
        ASSET_ID: asset_id,
        "side": side,
        "requested_quantity": requested,
        "order_quantity": executable,
        "filled_quantity": executable,
        "unfilled_quantity": requested - executable,
        "implementation_gap": (
            (requested - executable) / requested if requested else 0.0
        ),
        "status": status,
        "reason": reason or None,
        "expires_at": session if expire_unfilled and requested > executable else None,
    }


def _reverse_asset_id(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value)


def _rows_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        if rows
        else pl.DataFrame()
    )


__all__ = [
    "AccountBacktestConfig",
    "AccountBacktestResult",
    "AccountStateCheckpoint",
    "run_account_backtest",
    "run_planned_account_backtest",
]
