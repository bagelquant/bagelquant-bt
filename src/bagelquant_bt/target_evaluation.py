"""Evaluation of stored target snapshots; no prediction or optimizer builds."""

from __future__ import annotations

from dataclasses import replace

import polars as pl

from .account import (
    AccountBacktestConfig,
    AccountStateCheckpoint,
    PreparedAccountMarketData,
    StatefulAccountBacktestResult,
    run_stateful_account_backtest,
)
from .exceptions import InputValidationError


def implementation_stress_scenarios(config: AccountBacktestConfig, *, session_lag: int):
    """Yield one-factor scenarios with tax, minimum fee and account rules fixed.

    The capital scale also scales the fixed-notional capital target. Otherwise
    that policy would withdraw the extra initial cash at the first rebalance,
    leaving capital/capacity stress with the unchanged invested notional.
    """
    for multiple in (2, 5, 10):
        yield (
            f"capital_{multiple}",
            replace(
                config,
                initial_capital=config.initial_capital * multiple,
                fixed_notional=config.fixed_notional * multiple,
            ),
            session_lag,
        )
    for multiple in (2, 4):
        yield (
            f"commission_{multiple}",
            replace(
                config,
                transaction_cost=replace(
                    config.transaction_cost,
                    rate=config.transaction_cost.rate * multiple,
                ),
            ),
            session_lag,
        )
        yield (
            f"slippage_{multiple}",
            replace(
                config,
                transaction_cost=replace(
                    config.transaction_cost,
                    buy_slippage_rate=config.transaction_cost.buy_slippage_rate
                    * multiple,
                    sell_slippage_rate=config.transaction_cost.sell_slippage_rate
                    * multiple,
                ),
            ),
            session_lag,
        )
    for delay in (1, 4):
        yield f"delay_{delay}", config, session_lag + delay


def evaluate_portfolio_targets(
    weights: pl.DataFrame,
    decisions: pl.DataFrame,
    market_prices: pl.DataFrame | PreparedAccountMarketData,
    *,
    calendar: pl.DataFrame,
    corporate_action_coverage: pl.DataFrame,
    config: AccountBacktestConfig,
    session_lag: int = 1,
    execution_availability: pl.DataFrame | None = None,
    lot_sizes: pl.DataFrame | None = None,
    corporate_actions: pl.DataFrame | None = None,
    checkpoint: AccountStateCheckpoint | None = None,
    initial_target_position_plans: pl.DataFrame | None = None,
    historical_positions: pl.DataFrame | None = None,
    historical_sessions: pl.DataFrame | None = None,
) -> StatefulAccountBacktestResult:
    """Simulate explicit rebalance snapshots while retaining hold instructions.

    Account state determines lot sizing, cash and fills. It cannot change saved
    weights. The supplied calendar, prices and corporate-action coverage must
    span the simulated sessions. The calendar must also contain the known
    execution sessions after the last decision; prices still stop at the
    simulation cutoff. Preserve the returned future target-position plans
    when continuing, so their decision-close quantities are never resized.
    Checkpoint continuation is authorized by the
    caller only after proving that all historical inputs are unchanged.
    """
    if (
        isinstance(session_lag, bool)
        or not isinstance(session_lag, int)
        or session_lag < 1
    ):
        raise InputValidationError("session_lag must be a positive integer")
    if not {"time", "asset_id", "value"} <= set(weights.columns):
        raise InputValidationError("weights requires time, asset_id and value")
    if not {"time", "status", "reason"} <= set(decisions.columns):
        raise InputValidationError("decisions requires time, status and reason")
    if (
        decisions["time"].n_unique() != decisions.height
        or decisions["time"].null_count()
    ):
        raise InputValidationError("decision dates must be unique and non-null")
    if decisions.filter(
        ~pl.col("status").is_in(["rebalance", "hold", "unavailable"])
        | pl.col("status").is_null()
    ).height:
        raise InputValidationError("unknown decision status")
    if weights.unique(["time", "asset_id"]).height != weights.height:
        raise InputValidationError("target coordinates must be unique")
    if weights.filter(
        pl.col("value").is_null() | ~pl.col("value").is_finite() | (pl.col("value") < 0)
    ).height:
        raise InputValidationError("saved weights must be finite and nonnegative")
    rebalances = set(
        decisions.filter(pl.col("status") == "rebalance")["time"].to_list()
    )
    if set(weights["time"].to_list()) != rebalances:
        raise InputValidationError(
            "target snapshots must exactly match rebalance decisions"
        )
    if (
        weights.group_by("time")
        .agg(pl.col("value").sum())
        .filter(pl.col("value") > 1 + 1e-7)
        .height
    ):
        raise InputValidationError("target weights exceed full investment")
    sessions = (
        calendar.select(pl.col("time").cast(pl.Date))
        .unique()
        .sort("time")["time"]
        .to_list()
    )
    positions = {day: index for index, day in enumerate(sessions)}
    schedule = []
    for day in sorted(rebalances):
        if day not in positions:
            raise InputValidationError(f"decision {day} is absent from calendar")
        index = positions[day] + session_lag
        if index >= len(sessions):
            raise InputValidationError(
                f"calendar requires {session_lag} known execution sessions "
                f"after decision {day}"
            )
        if checkpoint is None or day > checkpoint.time:
            schedule.append({"decision_date": day, "execution_date": sessions[index]})
    schedule_frame = pl.DataFrame(
        schedule, schema={"decision_date": pl.Date, "execution_date": pl.Date}
    )
    # The account planner gives every prior holding absent from this complete
    # positive sleeve an explicit zero target. Unheld zero coordinates require
    # neither a quote nor a lot allocation. Keep all zeros in the saved value.
    snapshots = {
        group["time"][0]: group.filter(pl.col("value") > 0).select(
            "asset_id", pl.col("value").alias("weight")
        )
        for group in weights.partition_by("time", maintain_order=True)
    }

    def saved_snapshot(context):
        return snapshots[context.decision_date]

    return run_stateful_account_backtest(
        schedule_frame,
        market_prices,
        saved_snapshot,
        corporate_action_coverage=corporate_action_coverage,
        config=config,
        execution_availability=execution_availability,
        lot_sizes=lot_sizes,
        corporate_actions=corporate_actions,
        checkpoint=checkpoint,
        initial_target_position_plans=initial_target_position_plans,
        historical_positions=historical_positions,
        historical_sessions=historical_sessions,
    )
