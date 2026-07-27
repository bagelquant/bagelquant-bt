"""Backtest orchestration."""

from __future__ import annotations

import polars as pl

from .config import BacktestConfig
from .costs import turnover
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
from .results import BacktestResult, TransactionCostBreakdown
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
) -> BacktestResult:
    """Backtest a weight frame with a precomputed forward-return panel."""

    executable_weights = _expand_portfolio_weights(weights, prices, forward_returns)
    executable_weights, execution_blocks = _apply_execution_availability(
        executable_weights, execution_availability
    )
    if executable_weights.is_empty():
        raise InputValidationError("at least two overlapping price times are required")

    gross_returns = portfolio_returns(executable_weights, forward_returns)
    turn = turnover(executable_weights)
    costs, returns, value = _simulate_cost_adjusted_returns(
        weights=executable_weights,
        gross_returns=gross_returns,
        config=config,
    )
    summary, performance = summarize_performance(
        returns=returns,
        turnover=turn,
        costs=costs,
        initial_capital=config.initial_capital,
        annualization=config.annualization,
    )
    return BacktestResult(
        weights=executable_weights,
        asset_returns=forward_returns,
        returns=returns,
        value=value,
        turnover=turn,
        transaction_costs=costs,
        summary=summary,
        performance=performance,
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
        execution_blocks=execution_blocks,
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
) -> tuple[pl.DataFrame, pl.DataFrame]:
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
    if execution_availability is None:
        return desired_weights, empty
    availability = validate_execution_availability(execution_availability)
    rules = {
        (row[TIME], row[ASSET_ID]): row
        for row in availability.iter_rows(named=True)
    }
    executed: dict[str, float] = {}
    weight_rows: list[dict[str, object]] = []
    blocked_rows: list[dict[str, object]] = []
    for row in desired_weights.sort([TIME, ASSET_ID]).iter_rows(named=True):
        time = row[TIME]
        asset_id = str(row[ASSET_ID])
        target = float(row["weight"])
        retained = executed.get(asset_id, 0.0)
        delta = target - retained
        rule = rules.get((time, asset_id))
        blocked_side = None
        if rule is not None and delta > 0.0 and not bool(rule["can_buy"]):
            blocked_side = "buy"
        elif rule is not None and delta < 0.0 and not bool(rule["can_sell"]):
            blocked_side = "sell"
        if blocked_side is not None:
            blocked_rows.append(
                {
                    TIME: time,
                    ASSET_ID: asset_id,
                    "side": blocked_side,
                    "target_weight": target,
                    "retained_weight": retained,
                    "reason": str(rule["reason"]),
                }
            )
            resolved = retained
        else:
            resolved = target
            executed[asset_id] = resolved
        weight_rows.append(
            {TIME: time, ASSET_ID: asset_id, "weight": resolved}
        )
    return (
        pl.DataFrame(
            weight_rows,
            schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64},
        ).sort([TIME, ASSET_ID]),
        (
            pl.DataFrame(blocked_rows, schema=empty.schema)
            if blocked_rows
            else empty
        ),
    )


def _simulate_cost_adjusted_returns(
    *,
    weights: pl.DataFrame,
    gross_returns: pl.DataFrame,
    config: BacktestConfig,
) -> tuple[TransactionCostBreakdown, pl.DataFrame, pl.DataFrame]:
    trade_summary = _trade_summary(weights)
    gross_by_time = {
        row[TIME]: float(row["gross_return"] or 0.0)
        for row in gross_returns.iter_rows(named=True)
    }

    current_gross_value = float(config.initial_capital)
    current_net_value = float(config.initial_capital)
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


def _trade_summary(weights: pl.DataFrame) -> pl.DataFrame:
    return (
        weights.sort([ASSET_ID, TIME])
        .with_columns(
            pl.col("weight")
            .fill_null(0.0)
            .shift(1)
            .over(ASSET_ID)
            .fill_null(0.0)
            .alias("previous_weight")
        )
        .with_columns(
            (pl.col("weight").fill_null(0.0) - pl.col("previous_weight"))
            .abs()
            .alias("weight_delta")
        )
        .group_by(TIME)
        .agg(
            pl.col("weight_delta")
            .filter(pl.col("weight_delta") > 0.0)
            .alias("weight_deltas"),
        )
        .sort(TIME)
    )


def _require_config(config: BacktestConfig | None) -> BacktestConfig:
    if config is None:
        raise BacktestConfigError(
            "config is required because initial_capital is needed for minimum fees"
        )
    return config
