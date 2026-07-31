"""Backtest orchestration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl

from .config import BacktestConfig
from .costs import _sparse_weight_deltas, _turnover_from_weight_deltas
from .exceptions import BacktestConfigError, InputValidationError
from .inputs import (
    ASSET_ID,
    TIME,
    asset_coverage,
    missing_price_keys,
    validate_execution_availability,
    validate_prices,
    validate_weights,
)
from .performance import summarize_performance
from .results import (
    BacktestResult,
    PerformanceSummary,
    TransactionCostBreakdown,
)
from .returns import (
    _expand_portfolio_weight_data,
    _portfolio_targets,
    _prepare_price_data,
    portfolio_returns,
    unexecuted_weight_keys,
)


def run_weight_backtest(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig | None = None,
    execution_availability: pl.DataFrame | None = None,
) -> BacktestResult:
    """Backtest a long-form portfolio weight frame."""

    resolved_config = _require_config(config)
    aligned_weights = validate_weights(weights)
    aligned_prices = validate_prices(prices)
    return backtest_weight_frame(
        aligned_weights,
        aligned_prices,
        config=resolved_config,
        execution_availability=execution_availability,
    )


def backtest_weight_frame(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None = None,
) -> BacktestResult:
    """Backtest an already materialized long-form weight frame."""

    aligned_weights = validate_weights(weights)
    aligned_prices = validate_prices(prices)
    price_data = _prepare_price_data(aligned_prices, inputs_sorted=True)
    missing_keys = missing_price_keys(aligned_weights, aligned_prices)
    return _backtest_weight_frame_with_forward_returns(
        aligned_weights,
        aligned_prices,
        price_data.forward_returns,
        config=config,
        missing_price_keys=missing_keys,
        price_gaps=price_data.price_gaps,
        execution_availability=execution_availability,
    )


def _backtest_weight_frame_with_forward_returns(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    missing_price_keys: pl.DataFrame | None = None,
    price_gaps: pl.DataFrame | None = None,
    execution_availability: pl.DataFrame | None = None,
    initial_target_weights: pl.DataFrame | None = None,
    initial_executed_weights: pl.DataFrame | None = None,
    initial_gross_value: float | None = None,
    initial_net_value: float | None = None,
    prepared_active_assets: pl.DataFrame | None = None,
    execution_availability_validated: bool = False,
) -> BacktestResult:
    """Backtest a weight frame with a precomputed forward-return panel."""

    (
        active_prices,
        active_returns,
        active_availability,
    ) = _active_market_inputs(
        weights,
        prices,
        forward_returns,
        execution_availability,
        initial_target_weights=initial_target_weights,
        initial_executed_weights=initial_executed_weights,
        prepared_active_assets=prepared_active_assets,
    )
    core = _run_weight_backtest_core(
        weights,
        active_prices,
        active_returns,
        config=config,
        execution_availability=active_availability,
        initial_target_weights=initial_target_weights,
        initial_executed_weights=initial_executed_weights,
        initial_gross_value=initial_gross_value,
        initial_net_value=initial_net_value,
        execution_availability_validated=execution_availability_validated,
    )
    return BacktestResult(
        weights=core.weights,
        asset_returns=forward_returns,
        returns=core.returns,
        value=core.value,
        turnover=core.turnover,
        transaction_costs=core.costs,
        summary=core.summary,
        performance=core.performance,
        annualization=config.annualization,
        coverage=asset_coverage(
            weights,
            prices,
            asset_count_column="weight_asset_count",
        ),
        missing_price_keys=(
            _empty_missing_price_keys()
            if missing_price_keys is None
            else missing_price_keys.sort([TIME, ASSET_ID])
        ),
        price_gaps=(
            _empty_price_gaps()
            if price_gaps is None
            else price_gaps.sort([TIME, ASSET_ID])
        ),
        unexecuted_weight_keys=unexecuted_weight_keys(weights, prices),
        execution_blocks=core.execution_blocks,
        target_weights=core.target_weights,
        execution_event_count=core.execution_event_count,
    )


@dataclass(frozen=True, slots=True)
class _CompactBacktestResult:
    """Return only the paths consumed by aggregate factor diagnostics."""

    returns: pl.DataFrame
    value: pl.DataFrame
    summary: PerformanceSummary


@dataclass(frozen=True, slots=True)
class _BacktestCoreResult:
    target_weights: pl.DataFrame
    weights: pl.DataFrame
    returns: pl.DataFrame
    value: pl.DataFrame
    turnover: pl.DataFrame
    costs: TransactionCostBreakdown
    summary: PerformanceSummary
    performance: pl.DataFrame
    execution_blocks: pl.DataFrame
    execution_event_count: int


def _compact_backtest_weight_frame_with_forward_returns(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None = None,
    execution_availability_validated: bool = False,
) -> _CompactBacktestResult:
    """Run a backtest without building diagnostics discarded by the caller."""

    (
        active_prices,
        active_returns,
        active_availability,
    ) = _active_market_inputs(
        weights,
        prices,
        forward_returns,
        execution_availability,
    )
    return _compact_backtest_weight_frame_with_active_market(
        weights,
        active_prices,
        active_returns,
        config=config,
        execution_availability=active_availability,
        execution_availability_validated=execution_availability_validated,
    )


def _compact_backtest_weight_frame_with_active_market(
    weights: pl.DataFrame,
    active_prices: pl.DataFrame,
    active_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None,
    execution_availability_validated: bool,
    execution_keys: pl.DataFrame | None = None,
) -> _CompactBacktestResult:
    """Run a compact backtest with caller-reused active market inputs."""

    return _run_sparse_compact_backtest(
        weights,
        active_prices,
        active_returns,
        config=config,
        execution_availability=execution_availability,
        execution_availability_validated=execution_availability_validated,
        execution_keys=execution_keys,
    )


def _run_sparse_compact_backtest(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None,
    execution_availability_validated: bool,
    execution_keys: pl.DataFrame | None,
) -> _CompactBacktestResult:
    """Compute aggregate paths from sparse holding-state changes."""

    target_events = _portfolio_targets(
        weights,
        prices,
        include_events=True,
    ).events
    if target_events.is_empty():
        raise InputValidationError("at least two overlapping price times are required")
    resolved_availability = (
        execution_availability
        if execution_availability is None or execution_availability_validated
        else validate_execution_availability(execution_availability)
    )
    market_keys = (
        forward_returns.select(TIME, ASSET_ID)
        .unique()
        .sort([ASSET_ID, TIME])
        if execution_keys is None
        else execution_keys
    )
    sparse_desired = _sparse_execution_desired_weights(
        target_events,
        market_keys,
        resolved_availability,
    )
    executable_events, _, _ = _apply_execution_availability(
        sparse_desired,
        resolved_availability,
        retry_blocked=config.retry_blocked_orders,
        availability_validated=True,
        target_events=target_events,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        held_returns = (
            forward_returns.sort([ASSET_ID, TIME])
            .join_asof(
                executable_events.sort([ASSET_ID, TIME]),
                on=TIME,
                by=ASSET_ID,
                strategy="backward",
            )
            .filter(pl.col("weight").is_not_null() & (pl.col("weight") != 0.0))
        )
    first_time = weights.get_column(TIME).min()
    gross_returns = (
        forward_returns.filter(pl.col(TIME) >= first_time)
        .select(TIME)
        .unique()
        .join(
            held_returns.with_columns(
                (pl.col("weight") * pl.col("forward_return")).alias(
                    "_weighted_return"
                )
            )
            .group_by(TIME)
            .agg(pl.col("_weighted_return").sum().alias("gross_return")),
            on=TIME,
            how="left",
        )
        .with_columns(pl.col("gross_return").fill_null(0.0))
        .sort(TIME)
    )
    deltas = _sparse_weight_deltas(executable_events, initial_weights=None)
    turn = (
        gross_returns.select(TIME)
        .join(_turnover_from_weight_deltas(deltas), on=TIME, how="left")
        .with_columns(pl.col("turnover").fill_null(0.0))
    )
    costs, returns, value = _simulate_cost_adjusted_returns(
        deltas=deltas,
        gross_returns=gross_returns,
        config=config,
    )
    summary, _ = summarize_performance(
        returns=returns,
        turnover=turn,
        costs=costs,
        initial_capital=config.initial_capital,
        annualization=config.annualization,
    )
    return _CompactBacktestResult(returns=returns, value=value, summary=summary)


def _sparse_execution_desired_weights(
    target_events: pl.DataFrame,
    market_keys: pl.DataFrame,
    execution_availability: pl.DataFrame | None,
) -> pl.DataFrame:
    if execution_availability is None or execution_availability.is_empty():
        return target_events
    active_assets = target_events.select(ASSET_ID).unique()
    rule_keys = (
        execution_availability.select(TIME, ASSET_ID)
        .join(active_assets, on=ASSET_ID, how="inner")
        .join(market_keys, on=[TIME, ASSET_ID], how="inner")
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        after_rule_keys = (
            rule_keys.select(pl.col(TIME).alias("_rule_time"), ASSET_ID)
            .sort([ASSET_ID, "_rule_time"])
            .join_asof(
                market_keys.select(
                    pl.col(TIME).alias("_next_time"), ASSET_ID
                ),
                left_on="_rule_time",
                right_on="_next_time",
                by=ASSET_ID,
                strategy="forward",
                allow_exact_matches=False,
            )
            .drop_nulls("_next_time")
            .select(pl.col("_next_time").alias(TIME), ASSET_ID)
        )
        return (
            pl.concat(
                [
                    target_events.select(TIME, ASSET_ID),
                    rule_keys,
                    after_rule_keys,
                ]
            )
            .unique()
            .sort([ASSET_ID, TIME])
            .join_asof(
                target_events.sort([ASSET_ID, TIME]),
                on=TIME,
                by=ASSET_ID,
                strategy="backward",
            )
            .drop_nulls("weight")
            .sort([TIME, ASSET_ID])
        )


def _legacy_compact_backtest_weight_frame_with_active_market(
    weights: pl.DataFrame,
    active_prices: pl.DataFrame,
    active_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None,
    execution_availability_validated: bool,
    execution_keys: pl.DataFrame | None = None,
) -> _CompactBacktestResult:
    """Reference implementation retained for optimized-path regression tests."""

    core = _run_weight_backtest_core(
        weights,
        active_prices,
        active_returns,
        config=config,
        execution_availability=execution_availability,
        execution_availability_validated=execution_availability_validated,
        prepared_execution_keys=execution_keys,
    )
    return _CompactBacktestResult(
        returns=core.returns,
        value=core.value,
        summary=core.summary,
    )


def _active_market_inputs(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    execution_availability: pl.DataFrame | None,
    *,
    initial_target_weights: pl.DataFrame | None = None,
    initial_executed_weights: pl.DataFrame | None = None,
    prepared_active_assets: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """Restrict economic calculations to assets that can affect the portfolio."""

    active_assets = pl.concat(
        [
            weights.select(ASSET_ID),
            *(
                []
                if initial_target_weights is None
                else [initial_target_weights.select(ASSET_ID)]
            ),
            *(
                []
                if initial_executed_weights is None
                else [initial_executed_weights.select(ASSET_ID)]
            ),
            *(
                []
                if prepared_active_assets is None
                else [prepared_active_assets.select(ASSET_ID)]
            ),
        ]
    ).unique()
    market_assets = prices.select(ASSET_ID).unique()
    covers_market = (
        active_assets.height == market_assets.height
        and active_assets.join(market_assets, on=ASSET_ID, how="anti").is_empty()
    )
    if covers_market:
        return prices, forward_returns, execution_availability
    return (
        prices.join(active_assets, on=ASSET_ID, how="inner"),
        forward_returns.join(active_assets, on=ASSET_ID, how="inner"),
        (
            None
            if execution_availability is None
            else execution_availability.join(
                active_assets,
                on=ASSET_ID,
                how="inner",
            )
        ),
    )


def _run_weight_backtest_core(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None,
    initial_target_weights: pl.DataFrame | None = None,
    initial_executed_weights: pl.DataFrame | None = None,
    initial_gross_value: float | None = None,
    initial_net_value: float | None = None,
    execution_availability_validated: bool = False,
    prepared_execution_keys: pl.DataFrame | None = None,
) -> _BacktestCoreResult:
    expanded = _expand_portfolio_weight_data(
        weights,
        prices,
        forward_returns,
        initial_target_weights=initial_target_weights,
        include_target_events=(
            execution_availability is not None
            or initial_executed_weights is not None
        ),
        execution_keys=prepared_execution_keys,
    )
    target_weights = expanded.weights
    (
        executable_weights,
        execution_blocks,
        execution_event_count,
    ) = _apply_execution_availability(
        target_weights,
        execution_availability,
        initial_target_weights=initial_target_weights,
        initial_executed_weights=initial_executed_weights,
        retry_blocked=config.retry_blocked_orders,
        availability_validated=execution_availability_validated,
        target_events=expanded.target_events,
    )
    if executable_weights.is_empty():
        raise InputValidationError("at least two overlapping price times are required")

    gross_returns = portfolio_returns(executable_weights, forward_returns)
    deltas = _sparse_weight_deltas(
        executable_weights,
        initial_weights=initial_executed_weights,
    )
    turn = (
        gross_returns.select(TIME)
        .join(_turnover_from_weight_deltas(deltas), on=TIME, how="left")
        .with_columns(pl.col("turnover").fill_null(0.0))
    )
    costs, returns, value = _simulate_cost_adjusted_returns(
        deltas=deltas,
        gross_returns=gross_returns,
        config=config,
        initial_gross_value=initial_gross_value,
        initial_net_value=initial_net_value,
    )
    summary, performance = summarize_performance(
        returns=returns,
        turnover=turn,
        costs=costs,
        initial_capital=config.initial_capital,
        annualization=config.annualization,
    )
    return _BacktestCoreResult(
        target_weights=target_weights,
        weights=executable_weights,
        returns=returns,
        value=value,
        turnover=turn,
        costs=costs,
        summary=summary,
        performance=performance,
        execution_blocks=execution_blocks,
        execution_event_count=execution_event_count,
    )


def _empty_missing_price_keys() -> pl.DataFrame:
    return pl.DataFrame(schema={TIME: pl.Date, ASSET_ID: pl.String})


def _empty_price_gaps() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            TIME: pl.Date,
            ASSET_ID: pl.String,
            "last_observed_time": pl.Date,
            "missing_session_count": pl.Int64,
        }
    )


def _apply_execution_availability(
    desired_weights: pl.DataFrame,
    execution_availability: pl.DataFrame | None,
    *,
    initial_target_weights: pl.DataFrame | None = None,
    initial_executed_weights: pl.DataFrame | None = None,
    retry_blocked: bool = True,
    availability_validated: bool = False,
    target_events: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Retain the last executed target while a caller-authored trade is blocked."""

    empty = pl.DataFrame(
        schema={
            TIME: pl.Date,
            ASSET_ID: pl.String,
            "side": pl.String,
            "target_weight": pl.Float64,
            "retained_weight": pl.Float64,
            "reason": pl.String,
        }
    )
    if (
        execution_availability is None
        and initial_executed_weights is None
    ):
        return desired_weights, empty, 0
    availability = (
        (
            execution_availability
            if availability_validated
            else validate_execution_availability(execution_availability)
        )
        if execution_availability is not None
        else pl.DataFrame(
            schema={
                TIME: pl.Date,
                ASSET_ID: pl.String,
                "can_buy": pl.Boolean,
                "can_sell": pl.Boolean,
                "reason": pl.String,
            }
        )
    )
    initial_targets = _checkpoint_weights(
        initial_target_weights,
        column="_initial_target",
    )
    initial_executed = _checkpoint_weights(
        initial_executed_weights,
        column="_initial_executed",
    )
    desired = (
        desired_weights.sort([TIME, ASSET_ID])
        .join(initial_targets, on=ASSET_ID, how="left")
        .join(initial_executed, on=ASSET_ID, how="left")
    )
    desired_asset_time = desired.sort([ASSET_ID, TIME])
    checkpoint_keys = (
        desired_asset_time.group_by(ASSET_ID, maintain_order=True)
        .first()
        .filter(
            pl.col("_initial_target").fill_null(0.0)
            != pl.col("_initial_executed").fill_null(0.0)
        )
        .select(TIME, ASSET_ID)
        .with_columns(pl.lit(False).alias("_target_changed"))
        if initial_target_weights is not None
        or initial_executed_weights is not None
        else desired.head(0)
        .select(TIME, ASSET_ID)
        .with_columns(pl.lit(False).alias("_target_changed"))
    )
    if target_events is None:
        change_keys = (
            desired_asset_time.with_columns(
                (
                    pl.col("weight")
                    != pl.col("weight")
                    .shift(1)
                    .over(ASSET_ID)
                    .fill_null(pl.col("_initial_target"))
                    .fill_null(0.0)
                ).alias("_target_changed")
            )
            .filter(pl.col("_target_changed"))
            .select(TIME, ASSET_ID, "_target_changed")
        )
    else:
        change_keys = target_events.select(TIME, ASSET_ID).with_columns(
            pl.lit(True).alias("_target_changed")
        )
    change_keys = pl.concat([change_keys, checkpoint_keys])
    rule_keys = availability.select(TIME, ASSET_ID).join(
        desired.select(TIME, ASSET_ID),
        on=[TIME, ASSET_ID],
        how="inner",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        after_rule_keys = (
            availability.select(
                pl.col(TIME).alias("_rule_time"),
                ASSET_ID,
            )
            .sort([ASSET_ID, "_rule_time"])
            .join_asof(
                desired_asset_time.select(
                    pl.col(TIME).alias("_next_time"),
                    ASSET_ID,
                ),
                left_on="_rule_time",
                right_on="_next_time",
                by=ASSET_ID,
                strategy="forward",
                allow_exact_matches=False,
            )
            .drop_nulls("_next_time")
            .select(pl.col("_next_time").alias(TIME), ASSET_ID)
        )
    event_keys = (
        pl.concat(
            [
                change_keys,
                rule_keys.with_columns(pl.lit(False).alias("_target_changed")),
                after_rule_keys.with_columns(
                    pl.lit(False).alias("_target_changed")
                ),
            ]
        )
        .group_by(TIME, ASSET_ID)
        .agg(pl.col("_target_changed").any())
    )
    events = (
        event_keys.join(
            desired.select(
                TIME,
                ASSET_ID,
                "weight",
                "_initial_target",
                "_initial_executed",
            ),
            on=[TIME, ASSET_ID],
            how="inner",
        )
        .join(
            availability.rename(
                {
                    "can_buy": "_can_buy",
                    "can_sell": "_can_sell",
                    "reason": "_reason",
                }
            ),
            on=[TIME, ASSET_ID],
            how="left",
        )
        .with_columns(pl.col("_reason").is_not_null().alias("_has_rule"))
        .sort([TIME, ASSET_ID])
    )
    if events.is_empty():
        return (
            desired_weights.sort([TIME, ASSET_ID]),
            empty,
            0,
        )

    asset_lookup = (
        desired.select(ASSET_ID)
        .unique()
        .sort(ASSET_ID)
        .with_row_index("_asset_index")
    )
    events = events.join(asset_lookup, on=ASSET_ID, how="left")
    initial_state = (
        asset_lookup.join(initial_executed, on=ASSET_ID, how="left")
        .with_columns(pl.col("_initial_executed").fill_null(0.0))
        .sort("_asset_index")
    )
    executed = initial_state.get_column("_initial_executed").to_numpy().copy()
    cancelled_targets = np.zeros(len(executed), dtype=np.bool_)

    asset_indices = events.get_column("_asset_index").to_numpy()
    targets = events.get_column("weight").to_numpy()
    target_changed = events.get_column("_target_changed").to_numpy()
    has_rule = events.get_column("_has_rule").to_numpy()
    can_buy = events.get_column("_can_buy").fill_null(True).to_numpy()
    can_sell = events.get_column("_can_sell").fill_null(True).to_numpy()
    resolved = np.empty(events.height, dtype=np.float64)
    retained_values = np.empty(events.height, dtype=np.float64)
    blocked_indices: list[int] = []
    blocked_sides: list[str] = []

    for position in range(events.height):
        asset_index = int(asset_indices[position])
        target = float(targets[position])
        retained = float(executed[asset_index])
        retained_values[position] = retained
        if bool(target_changed[position]):
            cancelled_targets[asset_index] = False
        delta = target - retained
        blocked_side = None
        if has_rule[position] and delta > 0.0 and not can_buy[position]:
            blocked_side = "buy"
        elif has_rule[position] and delta < 0.0 and not can_sell[position]:
            blocked_side = "sell"
        if blocked_side is not None:
            blocked_indices.append(position)
            blocked_sides.append(blocked_side)
            resolved[position] = retained
            if not retry_blocked:
                cancelled_targets[asset_index] = True
        elif (
            not retry_blocked
            and cancelled_targets[asset_index]
            and not target_changed[position]
        ):
            resolved[position] = retained
        else:
            resolved[position] = target
            executed[asset_index] = target

    resolved_events = events.select(TIME, ASSET_ID).with_columns(
        pl.Series("weight", resolved)
    ).sort([ASSET_ID, TIME])
    blocked = empty
    if blocked_indices:
        blocked_positions = pl.DataFrame(
            {"_event_index": blocked_indices},
            schema={"_event_index": pl.UInt32},
        )
        blocked_events = (
            events.with_row_index("_event_index")
            .join(blocked_positions, on="_event_index", how="inner")
            .sort("_event_index")
        )
        blocked = pl.DataFrame(
            {
                TIME: blocked_events.get_column(TIME),
                ASSET_ID: blocked_events.get_column(ASSET_ID),
                "side": blocked_sides,
                "target_weight": blocked_events.get_column("weight"),
                "retained_weight": retained_values[
                    np.asarray(blocked_indices, dtype=np.int64)
                ],
                "reason": blocked_events.get_column("_reason"),
            },
            schema=empty.schema,
        )
    mismatch_positions = np.flatnonzero(resolved != targets)
    ordered_desired = desired.select(TIME, ASSET_ID, "weight")
    if mismatch_positions.size == 0:
        resolved_weights = ordered_desired
    else:
        affected_asset_indices = np.unique(asset_indices[mismatch_positions])
        affected_assets = asset_lookup.filter(
            pl.col("_asset_index").is_in(affected_asset_indices)
        ).select(ASSET_ID)
        affected_desired = (
            ordered_desired.with_row_index("_row_index")
            .join(affected_assets, on=ASSET_ID, how="inner")
            .join(initial_executed, on=ASSET_ID, how="left")
            .sort([ASSET_ID, TIME])
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Sortedness of columns cannot be checked when 'by' groups provided"
                ),
                category=UserWarning,
            )
            affected_resolved = (
                affected_desired.join_asof(
                    resolved_events.join(
                        affected_assets,
                        on=ASSET_ID,
                        how="inner",
                    ),
                    on=TIME,
                    by=ASSET_ID,
                    strategy="backward",
                )
                .with_columns(
                    pl.col("weight_right")
                    .fill_null(pl.col("_initial_executed"))
                    .fill_null(0.0)
                    .alias("_resolved_weight")
                )
                .select("_row_index", "_resolved_weight")
            )
        resolved_values = ordered_desired.get_column("weight").to_numpy().copy()
        resolved_values[
            affected_resolved.get_column("_row_index").to_numpy()
        ] = affected_resolved.get_column("_resolved_weight").to_numpy()
        resolved_weights = ordered_desired.with_columns(
            pl.Series("weight", resolved_values)
        )
    return (
        resolved_weights,
        blocked,
        events.height,
    )


def _simulate_cost_adjusted_returns(
    *,
    deltas: pl.DataFrame,
    gross_returns: pl.DataFrame,
    config: BacktestConfig,
    initial_gross_value: float | None = None,
    initial_net_value: float | None = None,
) -> tuple[TransactionCostBreakdown, pl.DataFrame, pl.DataFrame]:
    trade_summary = _trade_summary(deltas)
    timeline = (
        gross_returns.join(trade_summary, on=TIME, how="left")
        .with_columns(pl.col("gross_return").fill_null(0.0))
        .sort(TIME)
    )

    current_gross_value = float(
        config.initial_capital
        if initial_gross_value is None
        else initial_gross_value
    )
    current_net_value = float(
        config.initial_capital
        if initial_net_value is None
        else initial_net_value
    )
    periods = timeline.height
    times = timeline.get_column(TIME)
    gross_values = timeline.get_column("gross_return").to_numpy()
    delta_lists = timeline.get_column("weight_deltas")
    traded_asset_counts = (
        delta_lists.list.len().fill_null(0).to_numpy().astype(np.int64)
    )
    flat_deltas = (
        delta_lists.explode(empty_as_null=True).drop_nulls().to_numpy()
    )
    traded_notionals = np.empty(periods, dtype=np.float64)
    raw_fees = np.empty(periods, dtype=np.float64)
    min_fee_adjustments = np.empty(periods, dtype=np.float64)
    total_fees = np.empty(periods, dtype=np.float64)
    cost_returns = np.empty(periods, dtype=np.float64)
    net_returns = np.empty(periods, dtype=np.float64)
    gross_value_path = np.empty(periods, dtype=np.float64)
    net_value_path = np.empty(periods, dtype=np.float64)

    offset = 0
    for position in range(periods):
        time = times[position]
        traded_asset_count = int(traded_asset_counts[position])
        weight_deltas = flat_deltas[offset : offset + traded_asset_count]
        offset += traded_asset_count
        weight_delta = float(np.sum(weight_deltas))
        traded_notional = weight_delta * current_net_value
        raw_fee = traded_notional * config.transaction_cost.rate
        total_fee = float(
            np.maximum(
                weight_deltas * current_net_value * config.transaction_cost.rate,
                config.transaction_cost.min_fee,
            ).sum()
        )

        cost_return = total_fee / current_net_value if current_net_value else 0.0
        gross_return = float(gross_values[position])
        net_return = gross_return - cost_return
        next_net_value = current_net_value * (1.0 + net_return)
        if next_net_value <= 0.0:
            raise InputValidationError(
                "net portfolio value became non-positive after transaction costs "
                f"at {time}: current_value={current_net_value:.6g}, "
                f"gross_return={gross_return:.6g}, "
                f"cost_return={cost_return:.6g}, "
                f"traded_asset_count={traded_asset_count}, "
                f"total_fee={total_fee:.6g}. "
                "Increase initial_capital or reduce traded universe/turnover."
            )
        current_gross_value *= 1.0 + gross_return
        current_net_value = next_net_value
        traded_notionals[position] = traded_notional
        raw_fees[position] = raw_fee
        min_fee_adjustments[position] = total_fee - raw_fee
        total_fees[position] = total_fee
        cost_returns[position] = cost_return
        net_returns[position] = net_return
        gross_value_path[position] = current_gross_value
        net_value_path[position] = current_net_value

    costs = pl.DataFrame(
        {
            TIME: times,
            "traded_asset_count": traded_asset_counts,
            "traded_notional": traded_notionals,
            "raw_fee": raw_fees,
            "min_fee_adjustment": min_fee_adjustments,
            "total_fee": total_fees,
            "cost_return": cost_returns,
        }
    )
    returns = pl.DataFrame(
        {
            TIME: times,
            "gross_return": gross_values,
            "net_return": net_returns,
        }
    )
    value = pl.DataFrame(
        {
            TIME: times,
            "gross_value": gross_value_path,
            "net_value": net_value_path,
            "gross_return_cumulative": np.cumprod(1.0 + gross_values) - 1.0,
            "net_return_cumulative": np.cumprod(1.0 + net_returns) - 1.0,
        }
    )
    return TransactionCostBreakdown(costs), returns, value


def _trade_summary(
    deltas: pl.DataFrame,
) -> pl.DataFrame:
    return (
        deltas.group_by(TIME)
        .agg(
            pl.col("weight_delta")
            .filter(pl.col("weight_delta") > 0.0)
            .alias("weight_deltas"),
        )
        .sort(TIME)
    )


def _checkpoint_weights(
    frame: pl.DataFrame | None,
    *,
    column: str,
) -> pl.DataFrame:
    if frame is None:
        return pl.DataFrame(schema={ASSET_ID: pl.String, column: pl.Float64})
    required = {ASSET_ID, "weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError(
            f"checkpoint weights is missing required columns: {missing}"
        )
    normalized = frame.select(
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("weight").cast(pl.Float64).alias(column),
    )
    if normalized.get_column(ASSET_ID).n_unique() != normalized.height:
        raise InputValidationError("checkpoint weights must be unique by asset_id")
    return normalized


def _require_config(config: BacktestConfig | None) -> BacktestConfig:
    if config is None:
        raise BacktestConfigError(
            "config is required because initial_capital is needed for minimum fees"
        )
    return config
