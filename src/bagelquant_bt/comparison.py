"""Total-return performance paths and execution-account reconciliation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from .account import AccountBacktestResult
from .config import TransactionCostConfig
from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME
from .performance import _annualized_return


@dataclass(frozen=True, slots=True)
class TotalReturnTargetPath:
    """Continuous target exposure valued by a total-return price index."""

    returns: pl.DataFrame
    annualization: int


@dataclass(frozen=True, slots=True)
class ActualPerformancePath:
    """Fill-authored actual exposure valued by a total-return price index."""

    returns: pl.DataFrame
    annualization: int


@dataclass(frozen=True, slots=True)
class PortfolioPathComparison:
    """Four performance paths plus the separate execution-account truth."""

    nav_paths: pl.DataFrame
    attribution: pl.DataFrame
    summary: pl.DataFrame
    reconciliation: pl.DataFrame


_PORTFOLIO_RETURN_COLUMNS = (
    "target_gross_return",
    "target_net_return",
    "actual_gross_return",
    "actual_net_return",
)


def run_continuous_target_path(
    target_weights: pl.DataFrame,
    total_return_prices: pl.DataFrame,
    *,
    initial_capital: float,
    transaction_cost: TransactionCostConfig,
    execution_availability: pl.DataFrame | None = None,
    annualization: int = 252,
) -> TotalReturnTargetPath:
    """Run execution-date targets on a total-return price index.

    ``total_return_prices`` is an index, not an executable market price. Its
    meaningful value is the ratio between dates. Corporate actions must not
    enter this boundary because they are already represented by the index.
    """

    paths = run_total_return_weight_paths(
        {"target": target_weights},
        total_return_prices,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        execution_availability=execution_availability,
    )
    return TotalReturnTargetPath(
        paths.filter(pl.col("portfolio") == "target")
        .select(
            TIME,
            pl.col("gross_return"),
            pl.col("net_return"),
        )
        .sort(TIME),
        annualization,
    )


def run_total_return_weight_paths(
    weight_frames: Mapping[str, pl.DataFrame],
    total_return_prices: pl.DataFrame,
    *,
    initial_capital: float = 1.0,
    transaction_cost: TransactionCostConfig | None = None,
    execution_availability: pl.DataFrame | None = None,
    retry_blocked: bool = True,
) -> pl.DataFrame:
    """Evaluate several sparse target frames as buy-and-hold unit portfolios.

    A target snapshot creates index units once.  Units then remain unchanged
    until another target snapshot (or an explicitly retried blocked order), so
    weights drift naturally between rebalances instead of being reset daily.
    """

    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise InputValidationError("initial capital must be finite and positive")
    prices = _total_return_prices(total_return_prices)
    if prices.is_empty():
        raise InputValidationError("total-return prices must not be empty")
    targets = {
        label: _target_lookup(frame)
        for label, frame in weight_frames.items()
    }
    if not targets:
        return _empty_labeled_returns()
    availability = _availability_lookup(execution_availability)
    states = {
        label: {
            "gross": _performance_state(initial_capital),
            "net": _performance_state(initial_capital),
            "pending_gross": {},
            "pending_net": {},
            "pending_weight_gross": {},
            "pending_weight_net": {},
            "started": False,
            "previous_gross": initial_capital,
            "previous_net": initial_capital,
        }
        for label in targets
    }
    rows: list[dict[str, Any]] = []
    for session, session_frame in prices.group_by(TIME, maintain_order=True):
        day = session[0]
        closing_marks = {
            str(row[ASSET_ID]): float(row["total_return_price"])
            for row in session_frame.iter_rows(named=True)
        }
        execution_marks = {
            str(row[ASSET_ID]): float(row["execution_total_return_price"])
            for row in session_frame.iter_rows(named=True)
        }
        for label, target_by_date in targets.items():
            state = states[label]
            for ledger in (state["gross"], state["net"]):
                _mark_performance_state(ledger, execution_marks)
            target = target_by_date.get(day)
            if target is not None:
                state["started"] = True
                (
                    state["pending_gross"],
                    state["pending_weight_gross"],
                ) = _rebalance_performance_target(
                    state["gross"],
                    target,
                    execution_marks,
                    day,
                    availability,
                    costs=None,
                    retry_blocked=retry_blocked,
                )
                (
                    state["pending_net"],
                    state["pending_weight_net"],
                ) = _rebalance_performance_target(
                    state["net"],
                    target,
                    execution_marks,
                    day,
                    availability,
                    costs=transaction_cost,
                    retry_blocked=retry_blocked,
                )
            elif retry_blocked:
                for ledger_name, pending_name, costs in (
                    ("gross", "pending_gross", None),
                    ("net", "pending_net", transaction_cost),
                ):
                    pending = state[pending_name]
                    weight_name = pending_name.replace("pending_", "pending_weight_")
                    pending_weights = state[weight_name]
                    if pending_weights:
                        equity = _performance_equity(state[ledger_name])
                        resolved = {
                            asset: weight * equity / execution_marks[asset]
                            for asset, weight in pending_weights.items()
                            if asset in execution_marks
                        }
                        pending = {**pending, **resolved}
                        state[weight_name] = {
                            asset: weight
                            for asset, weight in pending_weights.items()
                            if asset not in execution_marks
                        }
                    if pending:
                        state[pending_name] = _execute_pending_units(
                            state[ledger_name],
                            pending,
                            execution_marks,
                            day,
                            availability,
                            costs,
                        )
            if not state["started"]:
                continue
            for ledger in (state["gross"], state["net"]):
                _mark_performance_state(ledger, closing_marks)
            gross_equity = _performance_equity(state["gross"])
            net_equity = _performance_equity(state["net"])
            rows.append(
                {
                    TIME: day,
                    "portfolio": label,
                    "gross_return": gross_equity / state["previous_gross"] - 1.0,
                    "net_return": net_equity / state["previous_net"] - 1.0,
                }
            )
            state["previous_gross"] = gross_equity
            state["previous_net"] = net_equity
    return (
        pl.DataFrame(rows).sort(TIME, "portfolio")
        if rows
        else _empty_labeled_returns()
    )


def run_actual_performance_path(
    fills: pl.DataFrame,
    positions: pl.DataFrame,
    total_return_prices: pl.DataFrame,
    *,
    initial_capital: float,
    annualization: int = 252,
) -> ActualPerformancePath:
    """Value actual fill-authored exposure using total-return index changes.

    Raw quantities decide which exposure exists.  Total-return prices decide
    what that exposure earns.  Dividend cash and share events never enter this
    ledger; explicit fill costs are the only Gross-to-Net bridge.
    """

    if initial_capital <= 0 or annualization <= 0:
        raise InputValidationError("capital and annualization must be positive")
    prices = _total_return_prices(total_return_prices)
    validated_fills = _validated_fills(fills)
    position_lookup = _position_lookup(positions)
    fills_by_date = {
        key[0]: tuple(frame.sort("order_id").to_dicts())
        for key, frame in validated_fills.group_by(TIME, maintain_order=True)
    }
    gross = _performance_state(initial_capital)
    previous_gross = initial_capital
    previous_net = initial_capital
    rows: list[dict[str, Any]] = []
    for session, session_frame in prices.group_by(TIME, maintain_order=True):
        day = session[0]
        closing_marks = {
            str(row[ASSET_ID]): float(row["total_return_price"])
            for row in session_frame.iter_rows(named=True)
        }
        execution_marks = {
            str(row[ASSET_ID]): float(row["execution_total_return_price"])
            for row in session_frame.iter_rows(named=True)
        }
        _mark_performance_state(gross, execution_marks)
        day_fills = fills_by_date.get(day, ())
        if day_fills:
            end_positions = position_lookup.get(day, {})
            net_fill_quantity: dict[str, int] = {}
            for fill in day_fills:
                direction = 1 if fill["side"] == "buy" else -1
                asset = str(fill[ASSET_ID])
                net_fill_quantity[asset] = (
                    net_fill_quantity.get(asset, 0)
                    + direction * int(fill["quantity"])
                )
            raw_shares = {
                asset: int(end_positions.get(asset, 0)) - delta
                for asset, delta in net_fill_quantity.items()
            }
            for fill in day_fills:
                asset = str(fill[ASSET_ID])
                index_price = execution_marks.get(asset)
                if index_price is None or index_price <= 0:
                    raise InputValidationError(
                        f"fill has no total-return price at {day}: {asset}"
                    )
                quantity = int(fill["quantity"])
                raw_price = float(fill["open_price"])
                if fill["side"] == "buy":
                    units = quantity * raw_price / index_price
                    gross["units"][asset] = gross["units"].get(asset, 0.0) + units
                    gross["marks"][asset] = index_price
                    gross["cash"] -= units * index_price
                    raw_shares[asset] = raw_shares.get(asset, 0) + quantity
                else:
                    before = raw_shares.get(asset, 0)
                    if before <= 0 or quantity > before:
                        raise InputValidationError(
                            f"sell fill exceeds actual holdings at {day}: {asset}"
                        )
                    fraction = quantity / before
                    owned = float(gross["units"].get(asset, 0.0))
                    removed = owned * fraction
                    gross["units"][asset] = owned - removed
                    gross["cash"] += removed * index_price
                    raw_shares[asset] = before - quantity
                explicit_cost = (
                    float(fill["commission"])
                    + float(fill["stamp_tax"])
                    + float(fill["transfer_fee"])
                    + float(fill["slippage_cost"])
                )
        _mark_performance_state(gross, closing_marks)
        gross_equity = _performance_equity(gross)
        explicit_cost = sum(
            float(fill["commission"])
            + float(fill["stamp_tax"])
            + float(fill["transfer_fee"])
            + float(fill["slippage_cost"])
            for fill in day_fills
        )
        gross_return = gross_equity / previous_gross - 1.0
        explicit_cost_rate = explicit_cost / previous_net
        net_return = gross_return - explicit_cost_rate
        net_equity = previous_net * (1.0 + net_return)
        position_value = gross_equity - float(gross["cash"])
        rows.append(
            {
                TIME: day,
                "gross_return": gross_return,
                "net_return": net_return,
                "gross_equity": gross_equity,
                "net_equity": net_equity,
                "gross_cash": float(gross["cash"]),
                "net_cash": net_equity - position_value,
                "explicit_cost": explicit_cost,
                "explicit_cost_rate": explicit_cost_rate,
            }
        )
        previous_gross = gross_equity
        previous_net = net_equity
    return ActualPerformancePath(pl.DataFrame(rows).sort(TIME), annualization)


def compare_portfolio_paths(
    target: TotalReturnTargetPath,
    actual: ActualPerformancePath,
    execution_account: AccountBacktestResult,
    *,
    initial_capital: float,
    annualization: int | None = None,
) -> PortfolioPathComparison:
    """Align performance truth with the separately reported execution truth."""

    resolved_annualization = (
        target.annualization if annualization is None else annualization
    )
    if resolved_annualization <= 0 or initial_capital <= 0:
        raise InputValidationError("annualization and initial capital must be positive")
    target_returns = _named_returns(target.returns, "target")
    actual_returns = _named_returns(actual.returns, "actual")
    aligned = (
        target_returns.join(actual_returns, on=TIME, how="full", coalesce=True)
        .sort(TIME)
        .with_columns(
            pl.exclude(TIME).fill_null(0.0),
        )
    )
    nav_paths = _portfolio_nav_paths(aligned)
    attribution = aligned.select(
        TIME,
        (pl.col("target_net_return") - pl.col("target_gross_return")).alias(
            "target_cost_drag"
        ),
        (pl.col("actual_gross_return") - pl.col("target_gross_return")).alias(
            "implementation_position_drag"
        ),
        (pl.col("actual_net_return") - pl.col("actual_gross_return")).alias(
            "actual_cost_drag"
        ),
    )
    summary = _summarize_portfolio_nav_paths(
        nav_paths,
        annualization=resolved_annualization,
    )
    reconciliation = _execution_reconciliation(
        nav_paths,
        actual,
        execution_account,
        initial_capital=initial_capital,
    )
    return PortfolioPathComparison(nav_paths, attribution, summary, reconciliation)


def summarize_portfolio_path_returns(
    returns: pl.DataFrame,
    *,
    annualization: int,
) -> pl.DataFrame:
    """Summarize four Portfolio paths over the supplied return window.

    The input window is treated as a fresh performance interval for cumulative
    and annualized metrics.  Callers may therefore slice an immutable, continuous
    Portfolio path without rebuilding its Artifact.
    """

    if annualization <= 0:
        raise InputValidationError("annualization must be positive")
    nav_paths = _portfolio_nav_paths(returns)
    return _summarize_portfolio_nav_paths(
        nav_paths,
        annualization=annualization,
    )


def _portfolio_nav_paths(returns: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(returns, pl.DataFrame):
        raise InputValidationError("portfolio path returns must be a Polars DataFrame")
    required = {TIME, *_PORTFOLIO_RETURN_COLUMNS}
    missing = sorted(required - set(returns.columns))
    if missing:
        raise InputValidationError(
            f"portfolio path returns are missing columns: {', '.join(missing)}"
        )
    frame = returns.select(
        pl.col(TIME).cast(pl.Date, strict=False),
        *(pl.col(column).cast(pl.Float64) for column in _PORTFOLIO_RETURN_COLUMNS),
    ).sort(TIME)
    if frame.get_column(TIME).null_count():
        raise InputValidationError("portfolio path times must be valid dates")
    if frame.get_column(TIME).n_unique() != frame.height:
        raise InputValidationError("portfolio path times must be unique")
    _validate_finite(frame)
    return frame.with_columns(
        *[
            (pl.col(column) + 1.0)
            .cum_prod()
            .alias(column.replace("return", "nav"))
            for column in _PORTFOLIO_RETURN_COLUMNS
        ]
    )


def _summarize_portfolio_nav_paths(
    nav_paths: pl.DataFrame,
    *,
    annualization: int,
) -> pl.DataFrame:
    return pl.concat(
        [
            _path_summary(nav_paths, column, annualization)
            for column in _PORTFOLIO_RETURN_COLUMNS
        ],
        how="vertical",
    )


def _performance_state(capital: float) -> dict[str, Any]:
    return {"cash": float(capital), "units": {}, "marks": {}}


def _mark_performance_state(state: dict[str, Any], prices: Mapping[str, float]) -> None:
    for asset in state["units"]:
        if asset in prices:
            state["marks"][asset] = float(prices[asset])


def _performance_equity(state: Mapping[str, Any]) -> float:
    return float(state["cash"]) + sum(
        float(units) * float(state["marks"].get(asset, 0.0))
        for asset, units in state["units"].items()
    )


def _rebalance_performance_target(
    state: dict[str, Any],
    target: Mapping[str, float],
    marks: Mapping[str, float],
    session: date,
    availability: Mapping[tuple[date, str], tuple[bool, bool]],
    *,
    costs: TransactionCostConfig | None,
    retry_blocked: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    equity = _performance_equity(state)
    missing_weights = {
        asset: float(weight)
        for asset, weight in target.items()
        if abs(float(weight)) > 1e-12 and asset not in marks
    }
    desired = {
        asset: float(weight) * equity / marks[asset]
        for asset, weight in target.items()
        if asset in marks and marks[asset] > 0
    }
    desired.update(
        {asset: 0.0 for asset in state["units"] if asset not in target}
    )
    return (
        _execute_pending_units(
            state,
            desired,
            marks,
            session,
            availability,
            costs,
            retry_blocked=retry_blocked,
        ),
        missing_weights if retry_blocked else {},
    )


def _execute_pending_units(
    state: dict[str, Any],
    desired: Mapping[str, float],
    marks: Mapping[str, float],
    session: date,
    availability: Mapping[tuple[date, str], tuple[bool, bool]],
    costs: TransactionCostConfig | None,
    *,
    retry_blocked: bool = True,
) -> dict[str, float]:
    pending: dict[str, float] = {}
    for asset in sorted(desired):
        price = marks.get(asset)
        if price is None:
            if retry_blocked:
                pending[asset] = float(desired[asset])
            continue
        current = float(state["units"].get(asset, 0.0))
        target = float(desired[asset])
        delta = target - current
        can_buy, can_sell = availability.get((session, asset), (True, True))
        blocked = (delta > 0 and not can_buy) or (delta < 0 and not can_sell)
        if blocked:
            if retry_blocked:
                pending[asset] = target
            continue
        notional = abs(delta) * price
        state["cash"] -= delta * price
        state["units"][asset] = target
        state["marks"][asset] = price
        if costs is not None and notional > 1e-12:
            side = "buy" if delta > 0 else "sell"
            state["cash"] -= (
                notional * costs.slippage_for(side)
                + max(costs.min_fee, notional * costs.rate)
                + notional * costs.transfer_fee_rate
                + (notional * costs.stamp_tax_rate if side == "sell" else 0.0)
            )
    return pending


def _total_return_prices(frame: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise InputValidationError("total_return_prices must be a polars DataFrame")
    value_column = next(
        (
            name
            for name in ("total_return_price", "adjusted_return_price", "price")
            if name in frame.columns
        ),
        None,
    )
    if value_column is None or not {TIME, ASSET_ID}.issubset(frame.columns):
        raise InputValidationError(
            "total_return_prices require (time, asset_id, total_return_price)"
        )
    execution_column = (
        "execution_total_return_price"
        if "execution_total_return_price" in frame.columns
        else value_column
    )
    result = (
        frame.select(
            pl.col(TIME).cast(pl.Date, strict=False),
            pl.col(ASSET_ID).cast(pl.String),
            pl.col(value_column).cast(pl.Float64).alias("total_return_price"),
            pl.col(execution_column)
            .cast(pl.Float64)
            .alias("execution_total_return_price"),
        )
        .drop_nulls()
        .sort(TIME, ASSET_ID)
    )
    if result.select(pl.struct(TIME, ASSET_ID).is_duplicated().any()).item():
        raise InputValidationError("total-return prices must be unique by key")
    if result.filter(
        (pl.col("total_return_price") <= 0)
        | (pl.col("execution_total_return_price") <= 0)
    ).height:
        raise InputValidationError("total-return prices must be positive")
    return result


def _target_lookup(frame: pl.DataFrame) -> dict[date, dict[str, float]]:
    column = "weight" if "weight" in frame.columns else "value"
    required = {TIME, ASSET_ID, column}
    if not required.issubset(frame.columns):
        raise InputValidationError("target weights require time, asset_id, and weight")
    selected = frame.select(
        pl.col(TIME).cast(pl.Date),
        pl.col(ASSET_ID).cast(pl.String),
        pl.col(column).cast(pl.Float64).alias("weight"),
    ).sort(TIME, ASSET_ID)
    if selected.select(pl.struct(TIME, ASSET_ID).is_duplicated().any()).item():
        raise InputValidationError("target weights must be unique by key")
    return {
        key[0]: {str(row[ASSET_ID]): float(row["weight"]) for row in group.to_dicts()}
        for key, group in selected.group_by(TIME, maintain_order=True)
    }


def _availability_lookup(
    frame: pl.DataFrame | None,
) -> dict[tuple[date, str], tuple[bool, bool]]:
    if frame is None or frame.is_empty():
        return {}
    required = {TIME, ASSET_ID, "can_buy", "can_sell"}
    if not required.issubset(frame.columns):
        raise InputValidationError("execution availability is missing required columns")
    return {
        (row[TIME], str(row[ASSET_ID])): (bool(row["can_buy"]), bool(row["can_sell"]))
        for row in frame.select(*required).iter_rows(named=True)
    }


def _validated_fills(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        TIME,
        "order_id",
        ASSET_ID,
        "side",
        "quantity",
        "open_price",
        "commission",
        "stamp_tax",
        "transfer_fee",
        "slippage_cost",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError(f"fills are missing columns: {missing}")
    return frame.select(*sorted(required)).with_columns(
        pl.col(TIME).cast(pl.Date),
        pl.col(ASSET_ID).cast(pl.String),
    ).sort(TIME, "order_id")


def _position_lookup(frame: pl.DataFrame) -> dict[date, dict[str, int]]:
    required = {TIME, ASSET_ID, "quantity"}
    if not required.issubset(frame.columns):
        raise InputValidationError("positions require time, asset_id, and quantity")
    return {
        key[0]: {str(row[ASSET_ID]): int(row["quantity"]) for row in group.to_dicts()}
        for key, group in frame.select(*required).sort(TIME, ASSET_ID).group_by(
            TIME, maintain_order=True
        )
    }


def _named_returns(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    required = {TIME, "gross_return", "net_return"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError(f"{prefix} returns are missing columns: {missing}")
    result = frame.select(
        TIME,
        pl.col("gross_return").cast(pl.Float64).alias(f"{prefix}_gross_return"),
        pl.col("net_return").cast(pl.Float64).alias(f"{prefix}_net_return"),
    ).sort(TIME)
    if result.get_column(TIME).n_unique() != result.height:
        raise InputValidationError(f"{prefix} returns must contain one row per time")
    return result


def _execution_reconciliation(
    nav_paths: pl.DataFrame,
    actual: ActualPerformancePath,
    account: AccountBacktestResult,
    *,
    initial_capital: float,
) -> pl.DataFrame:
    execution = account.account_value.select(
        TIME,
        "cash",
        "position_value",
        "cash_receivable",
        "stock_receivable_value",
        pl.col("equity").alias("execution_equity"),
        (pl.col("equity") / initial_capital).alias("execution_equity_index"),
    )
    performance_state = actual.returns.select(
        TIME,
        "net_equity",
        "net_cash",
        (pl.col("net_equity") - pl.col("net_cash")).alias(
            "performance_position_value"
        ),
    )
    return (
        nav_paths.select(TIME, pl.col("actual_net_nav").alias("performance_nav"))
        .join(execution, on=TIME, how="left")
        .join(performance_state, on=TIME, how="left")
        .with_columns(
            pl.col("execution_equity_index").forward_fill().fill_null(1.0),
            pl.col("cash").forward_fill().fill_null(initial_capital),
            pl.col("position_value").forward_fill().fill_null(0.0),
            pl.col("cash_receivable").fill_null(0.0),
            pl.col("stock_receivable_value").fill_null(0.0),
            pl.col("net_cash").forward_fill().fill_null(initial_capital),
            pl.col("performance_position_value").forward_fill().fill_null(0.0),
        )
        .with_columns(
            (pl.col("execution_equity_index") - pl.col("performance_nav")).alias(
                "total_gap"
            ),
            (pl.col("cash_receivable") / initial_capital).alias(
                "dividend_cash_effect"
            ),
            (pl.col("stock_receivable_value") / initial_capital).alias(
                "share_action_effect"
            ),
            pl.lit(0.0).alias("corporate_action_timing_effect"),
            pl.lit(0.0).alias("lot_rounding_effect"),
            ((pl.col("cash") - pl.col("net_cash")) / initial_capital).alias(
                "residual_cash_effect"
            ),
            pl.lit(0.0).alias("settlement_timing_effect"),
            pl.lit(0.0).alias("fee_accounting_effect"),
            (
                (pl.col("position_value") - pl.col("performance_position_value"))
                / initial_capital
            ).alias("position_valuation_effect"),
        )
        .with_columns(
            (
                pl.col("total_gap")
                - pl.sum_horizontal(
                    "dividend_cash_effect",
                    "share_action_effect",
                    "corporate_action_timing_effect",
                    "lot_rounding_effect",
                    "residual_cash_effect",
                    "settlement_timing_effect",
                    "fee_accounting_effect",
                    "position_valuation_effect",
                )
            ).alias("unexplained_gap")
        )
        .select(
            TIME,
            "performance_nav",
            "execution_equity",
            "execution_equity_index",
            "total_gap",
            "dividend_cash_effect",
            "share_action_effect",
            "corporate_action_timing_effect",
            "lot_rounding_effect",
            "residual_cash_effect",
            "settlement_timing_effect",
            "fee_accounting_effect",
            "position_valuation_effect",
            "unexplained_gap",
        )
        .sort(TIME)
    )


def _validate_finite(frame: pl.DataFrame) -> None:
    invalid = frame.select(
        pl.any_horizontal(
            *(
                pl.col(name).is_null() | ~pl.col(name).is_finite()
                for name in frame.columns
                if name != TIME
            )
        ).any()
    ).item()
    if invalid:
        raise InputValidationError("portfolio path returns must be finite")


def _path_summary(
    paths: pl.DataFrame,
    return_column: str,
    annualization: int,
) -> pl.DataFrame:
    values = paths.get_column(return_column).to_numpy()
    nav = paths.get_column(return_column.replace("return", "nav")).to_numpy()
    periods = len(values)
    total_return = float(nav[-1] - 1.0) if periods else math.nan
    annualized_return = _annualized_return(
        float(nav[-1]) if periods else math.nan,
        periods=periods,
        annualization=annualization,
    )
    volatility = (
        float(np.std(values, ddof=1) * math.sqrt(annualization))
        if periods > 1
        else math.nan
    )
    std = float(np.std(values, ddof=1)) if periods > 1 else math.nan
    sharpe = (
        float(np.mean(values) / std * math.sqrt(annualization))
        if std and math.isfinite(std)
        else math.nan
    )
    peak = np.maximum.accumulate(nav) if periods else np.array([], dtype=float)
    max_drawdown = float(np.min(nav / peak - 1.0)) if periods else math.nan
    return pl.DataFrame(
        {
            "path": [return_column.removesuffix("_return")],
            "cumulative_return": [total_return],
            "annualized_return": [annualized_return],
            "annualized_volatility": [volatility],
            "sharpe": [sharpe],
            "max_drawdown": [max_drawdown],
        }
    )


def _empty_labeled_returns() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            TIME: pl.Date,
            "portfolio": pl.String,
            "gross_return": pl.Float64,
            "net_return": pl.Float64,
        }
    )
