"""Backtest orchestration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import polars as pl

from .config import BacktestConfig
from .costs import turnover, weight_deltas
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
    _expand_portfolio_weights,
    cumulative_returns,
    portfolio_returns,
    prepare_price_data,
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
    price_data = prepare_price_data(aligned_prices)
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
    )
    published_weights = (
        _expand_portfolio_weights(
            weights,
            prices,
            forward_returns,
            initial_target_weights=initial_target_weights,
        )
        .join(
            core.weights.rename({"weight": "_resolved_weight"}),
            on=[TIME, ASSET_ID],
            how="left",
        )
        .with_columns(
            pl.col("_resolved_weight").fill_null(pl.col("weight")).alias("weight")
        )
        .drop("_resolved_weight")
        .sort([TIME, ASSET_ID])
    )
    return BacktestResult(
        weights=published_weights,
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
    core = _run_weight_backtest_core(
        weights,
        active_prices,
        active_returns,
        config=config,
        execution_availability=active_availability,
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
) -> _BacktestCoreResult:
    target_weights = _expand_portfolio_weights(
        weights,
        prices,
        forward_returns,
        initial_target_weights=initial_target_weights,
    )
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
    )
    if executable_weights.is_empty():
        raise InputValidationError("at least two overlapping price times are required")

    gross_returns = portfolio_returns(executable_weights, forward_returns)
    turn = turnover(
        executable_weights,
        initial_weights=initial_executed_weights,
    )
    costs, returns, value = _simulate_cost_adjusted_returns(
        weights=executable_weights,
        gross_returns=gross_returns,
        config=config,
        initial_weights=initial_executed_weights,
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
        validate_execution_availability(execution_availability)
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
    events = (
        desired.join(
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
        .with_columns(
            (
                pl.col("weight")
                != pl.col("weight")
                .shift(1)
                .over(ASSET_ID)
                .fill_null(pl.col("_initial_target"))
                .fill_null(0.0)
            ).alias("_target_changed"),
            pl.col("_reason").is_not_null().alias("_has_rule"),
            (
                pl.col("weight").cum_count().over(ASSET_ID) == 1
            ).alias("_first_row"),
        )
        .with_columns(
            pl.col("_has_rule")
            .shift(1)
            .over(ASSET_ID)
            .fill_null(False)
            .alias("_after_rule"),
            (
                pl.col("_first_row")
                & (
                    pl.col("_initial_target").fill_null(0.0)
                    != pl.col("_initial_executed").fill_null(0.0)
                )
            ).alias("_checkpoint_pending"),
        )
        .filter(
            pl.col("_target_changed")
            | pl.col("_has_rule")
            | pl.col("_after_rule")
            | pl.col("_checkpoint_pending")
        )
        .sort([TIME, ASSET_ID])
    )
    executed = {
        str(row[ASSET_ID]): float(row["_initial_executed"])
        for row in initial_executed.iter_rows(named=True)
    }
    event_rows: list[dict[str, object]] = []
    blocked_rows: list[dict[str, object]] = []
    cancelled_targets: dict[str, float] = {}
    for row in events.iter_rows(named=True):
        time = row[TIME]
        asset_id = str(row[ASSET_ID])
        target = float(row["weight"])
        retained = executed.get(asset_id, 0.0)
        if bool(row["_target_changed"]):
            cancelled_targets.pop(asset_id, None)
        delta = target - retained
        blocked_side = None
        if row["_has_rule"] and delta > 0.0 and not bool(row["_can_buy"]):
            blocked_side = "buy"
        elif row["_has_rule"] and delta < 0.0 and not bool(row["_can_sell"]):
            blocked_side = "sell"
        if blocked_side is not None:
            blocked_rows.append(
                {
                    TIME: time,
                    ASSET_ID: asset_id,
                    "side": blocked_side,
                    "target_weight": target,
                    "retained_weight": retained,
                    "reason": str(row["_reason"]),
                }
            )
            resolved = retained
            if not retry_blocked:
                cancelled_targets[asset_id] = target
        elif (
            not retry_blocked
            and asset_id in cancelled_targets
            and not bool(row["_target_changed"])
        ):
            resolved = retained
        else:
            resolved = target
            executed[asset_id] = resolved
        event_rows.append(
            {TIME: time, ASSET_ID: asset_id, "weight": resolved}
        )
    if not event_rows:
        return (
            desired_weights.sort([TIME, ASSET_ID]),
            empty,
            0,
        )
    resolved_events = pl.DataFrame(
        event_rows,
        schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64},
    ).sort([ASSET_ID, TIME])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        resolved_weights = (
            desired.select(TIME, ASSET_ID, "_initial_executed")
            .sort([ASSET_ID, TIME])
            .join_asof(
                resolved_events,
                on=TIME,
                by=ASSET_ID,
                strategy="backward",
            )
            .with_columns(
                pl.col("weight")
                .fill_null(pl.col("_initial_executed"))
                .fill_null(0.0)
                .alias("weight")
            )
            .drop("_initial_executed")
            .sort([TIME, ASSET_ID])
        )
    return (
        resolved_weights,
        (
            pl.DataFrame(blocked_rows, schema=empty.schema)
            if blocked_rows
            else empty
        ),
        events.height,
    )


def _simulate_cost_adjusted_returns(
    *,
    weights: pl.DataFrame,
    gross_returns: pl.DataFrame,
    config: BacktestConfig,
    initial_weights: pl.DataFrame | None = None,
    initial_gross_value: float | None = None,
    initial_net_value: float | None = None,
) -> tuple[TransactionCostBreakdown, pl.DataFrame, pl.DataFrame]:
    trade_summary = _trade_summary(
        weights,
        initial_weights=initial_weights,
    )
    gross_by_time = {
        row[TIME]: float(row["gross_return"] or 0.0)
        for row in gross_returns.iter_rows(named=True)
    }

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
    cost_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []

    for row in trade_summary.iter_rows(named=True):
        time = row[TIME]
        weight_deltas = [float(delta) for delta in row["weight_deltas"]]
        traded_asset_count = len(weight_deltas)
        weight_delta = sum(weight_deltas)
        traded_notional = weight_delta * current_net_value
        raw_fee = traded_notional * config.transaction_cost.rate
        total_fee = sum(
            max(
                delta * current_net_value * config.transaction_cost.rate,
                config.transaction_cost.min_fee,
            )
            for delta in weight_deltas
        )

        cost_return = total_fee / current_net_value if current_net_value else 0.0
        gross_return = gross_by_time.get(time, 0.0)
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
        cost_rows.append(
            {
                TIME: time,
                "traded_asset_count": traded_asset_count,
                "traded_notional": traded_notional,
                "raw_fee": raw_fee,
                "min_fee_adjustment": total_fee - raw_fee,
                "total_fee": total_fee,
                "cost_return": cost_return,
            }
        )
        return_rows.append(
            {TIME: time, "gross_return": gross_return, "net_return": net_return}
        )
        value_rows.append(
            {
                TIME: time,
                "gross_value": current_gross_value,
                "net_value": current_net_value,
            }
        )

    returns = pl.DataFrame(return_rows).sort(TIME)
    value = (
        pl.DataFrame(value_rows)
        .sort(TIME)
        .join(
            cumulative_returns(returns, "gross_return"),
            on=TIME,
        )
        .join(
            cumulative_returns(returns, "net_return"),
            on=TIME,
        )
    )
    return TransactionCostBreakdown(pl.DataFrame(cost_rows).sort(TIME)), returns, value


def _trade_summary(
    weights: pl.DataFrame,
    *,
    initial_weights: pl.DataFrame | None = None,
) -> pl.DataFrame:
    return (
        weight_deltas(weights, initial_weights=initial_weights)
        .group_by(TIME)
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
