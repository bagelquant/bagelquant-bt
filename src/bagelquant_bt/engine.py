"""Backtest orchestration."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace

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
    validate_slippage_rates,
    validate_weights,
)
from .performance import summarize_performance
from .results import (
    BacktestResult,
    PerformanceSummary,
    TransactionCostBreakdown,
    _DeferredMarketKeys,
    _DeferredPortfolioFrame,
)
from .returns import (
    _empty_unexecuted_weight_keys,
    _expand_portfolio_weight_data,
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
    slippage_rates: pl.DataFrame | None = None,
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
        slippage_rates=slippage_rates,
    )


def backtest_weight_frame(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
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
        slippage_rates=slippage_rates,
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
    slippage_rates: pl.DataFrame | None = None,
    initial_target_weights: pl.DataFrame | None = None,
    initial_executed_weights: pl.DataFrame | None = None,
    initial_gross_value: float | None = None,
    initial_net_value: float | None = None,
    initial_is_bankrupt: bool = False,
    prepared_active_assets: pl.DataFrame | None = None,
    execution_availability_validated: bool = False,
) -> BacktestResult:
    """Backtest a weight frame with a precomputed forward-return panel."""

    if (
        weights.is_empty()
        and initial_target_weights is None
        and initial_executed_weights is None
    ):
        raise InputValidationError(
            "weights must contain at least one target on the executable price calendar"
        )
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
    can_defer_weights = (
        initial_target_weights is None
        and initial_executed_weights is None
        and initial_gross_value is None
        and initial_net_value is None
        and not initial_is_bankrupt
    )
    resolved_slippage_rates = (
        None
        if slippage_rates is None
        else validate_slippage_rates(slippage_rates)
    )
    if can_defer_weights:
        execution_keys = (
            active_returns.select(TIME, ASSET_ID)
            .unique()
            .sort([ASSET_ID, TIME])
        )
        compact = _run_sparse_compact_backtests(
            {"portfolio": weights},
            active_prices,
            active_returns,
            config=config,
            execution_availability=active_availability,
            execution_availability_validated=execution_availability_validated,
            execution_keys=execution_keys,
            slippage_rates=resolved_slippage_rates,
        )["portfolio"]
        if compact.state is None:
            raise AssertionError("sparse portfolio state is required")
        deferred_market_keys = _DeferredMarketKeys(execution_keys)
        result_weights = _DeferredPortfolioFrame(
            market_keys=deferred_market_keys,
            state_events=compact.state.executable_events,
        )
        result_targets = _DeferredPortfolioFrame(
            market_keys=deferred_market_keys,
            state_events=compact.state.target_events,
        )
        result_returns = compact.returns
        result_value = compact.value
        result_turnover = compact.turnover
        result_costs = compact.costs
        result_summary = compact.summary
        result_performance = compact.performance
        result_blocks = compact.state.execution_blocks
        result_event_count = compact.state.execution_event_count
        result_unexecuted = compact.state.unexecuted_weight_keys
    else:
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
            initial_is_bankrupt=initial_is_bankrupt,
            execution_availability_validated=execution_availability_validated,
            slippage_rates=resolved_slippage_rates,
        )
        result_weights = core.weights
        result_targets = core.target_weights
        result_returns = core.returns
        result_value = core.value
        result_turnover = core.turnover
        result_costs = core.costs
        result_summary = core.summary
        result_performance = core.performance
        result_blocks = core.execution_blocks
        result_event_count = core.execution_event_count
        result_unexecuted = unexecuted_weight_keys(weights, prices)
    return BacktestResult(
        weights=result_weights,
        asset_returns=forward_returns,
        returns=result_returns,
        value=result_value,
        turnover=result_turnover,
        transaction_costs=result_costs,
        summary=result_summary,
        performance=result_performance,
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
        unexecuted_weight_keys=result_unexecuted,
        execution_blocks=result_blocks,
        target_weights=result_targets,
        execution_event_count=result_event_count,
    )


def _backtest_weight_frames_with_forward_returns(
    weight_frames: Mapping[str, pl.DataFrame],
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    price_gaps: pl.DataFrame | None,
    execution_availability: pl.DataFrame | None,
    execution_availability_validated: bool,
    slippage_rates: pl.DataFrame | None = None,
    market_context: _SparseMarketContext | None = None,
    precomputed_compact_results: Mapping[str, _CompactBacktestResult] | None = None,
) -> dict[str, BacktestResult]:
    """Build complete results for several portfolios from one batch scan."""

    nonempty = {
        label: weights
        for label, weights in weight_frames.items()
        if not weights.is_empty()
    }
    if not nonempty:
        return {}
    if market_context is None:
        combined_weights = pl.concat(list(nonempty.values()))
        active_prices, active_returns, active_availability = _active_market_inputs(
            combined_weights,
            prices,
            forward_returns,
            execution_availability,
        )
        execution_keys = (
            active_returns.select(TIME, ASSET_ID)
            .unique()
            .sort([ASSET_ID, TIME])
        )
    else:
        active_prices = prices
        active_returns = forward_returns
        active_availability = execution_availability
        execution_keys = market_context.execution_keys
    compact_results = (
        dict(precomputed_compact_results)
        if precomputed_compact_results is not None
        else _run_sparse_compact_backtests(
            nonempty,
            active_prices,
            active_returns,
            config=config,
            execution_availability=active_availability,
            execution_availability_validated=execution_availability_validated,
            execution_keys=execution_keys,
            market_context=market_context,
            slippage_rates=slippage_rates,
        )
    )
    results: dict[str, BacktestResult] = {}
    deferred_market_keys = _DeferredMarketKeys(execution_keys)
    for label, weights in nonempty.items():
        compact = compact_results[label]
        if compact.state is None:
            raise AssertionError("sparse portfolio state is required")
        results[label] = BacktestResult(
            weights=_DeferredPortfolioFrame(
                market_keys=deferred_market_keys,
                state_events=compact.state.executable_events,
            ),
            asset_returns=forward_returns,
            returns=compact.returns,
            value=compact.value,
            turnover=compact.turnover,
            transaction_costs=compact.costs,
            summary=compact.summary,
            performance=compact.performance,
            annualization=config.annualization,
            coverage=asset_coverage(
                weights,
                prices,
                asset_count_column="weight_asset_count",
            ),
            missing_price_keys=_empty_missing_price_keys(),
            price_gaps=(
                _empty_price_gaps()
                if price_gaps is None
                else price_gaps.sort([TIME, ASSET_ID])
            ),
            unexecuted_weight_keys=compact.state.unexecuted_weight_keys,
            execution_blocks=compact.state.execution_blocks,
            target_weights=_DeferredPortfolioFrame(
                market_keys=deferred_market_keys,
                state_events=compact.state.target_events,
            ),
            execution_event_count=compact.state.execution_event_count,
        )
    return results


@dataclass(frozen=True, slots=True)
class _CompactBacktestResult:
    """Return only the paths consumed by aggregate factor diagnostics."""

    returns: pl.DataFrame
    value: pl.DataFrame
    turnover: pl.DataFrame
    costs: TransactionCostBreakdown
    summary: PerformanceSummary
    performance: pl.DataFrame
    state: _SparsePortfolioState | None = None


@dataclass(frozen=True, slots=True)
class _SparsePortfolioState:
    """Sparse target and executed state changes for one portfolio."""

    target_events: pl.DataFrame
    executable_events: pl.DataFrame
    execution_blocks: pl.DataFrame
    execution_event_count: int
    first_time: object
    unexecuted_weight_keys: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _SparsePortfolioTargets:
    """Sparse state inputs plus absent-price rebalance diagnostics."""

    target_events: pl.DataFrame
    seed_targets: pl.DataFrame
    unexecuted_weight_keys: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _SparseMarketContext:
    """Read-only integer market representation reused by portfolio families."""

    market_times: pl.DataFrame
    market_assets: pl.DataFrame
    market_ordinals: np.ndarray
    asset_values: pl.Series
    asset_enum: pl.DataType
    observed: np.ndarray
    execution_keys: pl.DataFrame
    execution_event_keys: pl.DataFrame
    return_sessions: pl.DataFrame
    return_assets: pl.DataFrame
    return_matrix: np.ndarray


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


def _prepare_sparse_market_context(
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    execution_availability: pl.DataFrame | None,
    *,
    execution_keys: pl.DataFrame | None = None,
) -> _SparseMarketContext:
    """Encode stable market axes and rule events once for batch evaluation."""

    market_times = prices.get_column(TIME).unique(maintain_order=True).to_frame()
    market_assets = (
        prices.get_column(ASSET_ID).unique(maintain_order=True).to_frame()
    )
    market_ordinals = market_times.get_column(TIME).cast(pl.Int32).to_numpy()
    asset_values = market_assets.get_column(ASSET_ID)
    asset_enum = pl.Enum(asset_values.to_list())
    price_session_indices = np.searchsorted(
        market_ordinals,
        prices.get_column(TIME).cast(pl.Int32).to_numpy(),
    )
    price_asset_indices = (
        prices.get_column(ASSET_ID).cast(asset_enum).to_physical().to_numpy()
    )
    observed = np.zeros(
        (market_times.height, market_assets.height),
        dtype=np.bool_,
    )
    observed[price_session_indices, price_asset_indices] = True

    resolved_execution_keys = (
        forward_returns.select(TIME, ASSET_ID)
        if execution_keys is None
        else execution_keys
    )
    return_sessions = forward_returns.get_column(TIME).unique(
        maintain_order=True
    ).to_frame().with_row_index("_session_index")
    return_assets = (
        forward_returns.get_column(ASSET_ID)
        .unique(maintain_order=True)
        .to_frame()
        .with_row_index("_asset_index")
    )
    return_matrix = np.zeros(
        (return_sessions.height, return_assets.height),
        dtype=np.float64,
    )
    return_session_indices = np.searchsorted(
        return_sessions.get_column(TIME).cast(pl.Int32).to_numpy(),
        forward_returns.get_column(TIME).cast(pl.Int32).to_numpy(),
    )
    return_asset_indices = (
        forward_returns.get_column(ASSET_ID)
        .cast(pl.Enum(return_assets.get_column(ASSET_ID).to_list()))
        .to_physical()
        .to_numpy()
    )
    return_present = np.zeros(return_matrix.shape, dtype=np.bool_)
    return_matrix[
        return_session_indices,
        return_asset_indices,
    ] = forward_returns.get_column("forward_return").to_numpy()
    return_present[return_session_indices, return_asset_indices] = True
    execution_event_keys = _execution_rule_event_keys(
        execution_availability,
        return_sessions.get_column(TIME).cast(pl.Int32).to_numpy(),
        return_assets.get_column(ASSET_ID),
        return_present,
    )
    for values in (market_ordinals, observed, return_matrix):
        values.setflags(write=False)
    return _SparseMarketContext(
        market_times=market_times,
        market_assets=market_assets,
        market_ordinals=market_ordinals,
        asset_values=asset_values,
        asset_enum=asset_enum,
        observed=observed,
        execution_keys=resolved_execution_keys,
        execution_event_keys=execution_event_keys,
        return_sessions=return_sessions,
        return_assets=return_assets,
        return_matrix=return_matrix,
    )


def _compact_backtest_weight_frame_with_forward_returns(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None = None,
    execution_availability_validated: bool = False,
    slippage_rates: pl.DataFrame | None = None,
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
        slippage_rates=slippage_rates,
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
    slippage_rates: pl.DataFrame | None = None,
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
        slippage_rates=slippage_rates,
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
    slippage_rates: pl.DataFrame | None = None,
) -> _CompactBacktestResult:
    """Compute aggregate paths from sparse holding-state changes."""

    return _run_sparse_compact_backtests(
        {"portfolio": weights},
        prices,
        forward_returns,
        config=config,
        execution_availability=execution_availability,
        execution_availability_validated=execution_availability_validated,
        execution_keys=execution_keys,
        slippage_rates=slippage_rates,
    )["portfolio"]


def _run_sparse_compact_backtests(
    weight_frames: Mapping[str, pl.DataFrame],
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None,
    execution_availability_validated: bool,
    execution_keys: pl.DataFrame | None = None,
    market_context: _SparseMarketContext | None = None,
    slippage_rates: pl.DataFrame | None = None,
) -> dict[str, _CompactBacktestResult]:
    """Evaluate several sparse portfolios in one market-state scan."""

    nonempty = {
        label: weights
        for label, weights in weight_frames.items()
        if not weights.is_empty()
    }
    if not nonempty:
        return {}
    resolved_availability = (
        execution_availability
        if execution_availability is None or execution_availability_validated
        else validate_execution_availability(execution_availability)
    )
    resolved_slippage_rates = (
        None
        if slippage_rates is None
        else validate_slippage_rates(slippage_rates)
    )
    context = market_context or _prepare_sparse_market_context(
        prices,
        forward_returns,
        resolved_availability,
        execution_keys=execution_keys,
    )
    portfolio_targets = _batch_portfolio_targets(nonempty, context)
    states = {
        label: _resolve_sparse_portfolio_state_from_targets(
            portfolio_targets[label].target_events,
            portfolio_targets[label].seed_targets,
            resolved_availability,
            execution_event_keys=context.execution_event_keys,
            retry_blocked=config.retry_blocked_orders,
            first_time=weights.get_column(TIME).min(),
            unexecuted_weight_keys=portfolio_targets[label].unexecuted_weight_keys,
        )
        for label, weights in nonempty.items()
    }
    return _calculate_sparse_portfolio_batch(
        states,
        context,
        config=config,
        slippage_rates=resolved_slippage_rates,
    )


def _batch_portfolio_targets(
    weight_frames: Mapping[str, pl.DataFrame],
    market_context: _SparseMarketContext,
) -> dict[str, _SparsePortfolioTargets]:
    """Build target changes and first tradable states for all portfolios."""

    market_times = market_context.market_times
    market_assets = market_context.market_assets
    market_ordinals = market_context.market_ordinals
    asset_values = market_context.asset_values
    asset_enum = market_context.asset_enum
    observed = market_context.observed

    results: dict[str, _SparsePortfolioTargets] = {}
    for label, weights in weight_frames.items():
        timed_weights = weights.join(market_times, on=TIME, how="inner")
        snapshot_ordinals = (
            timed_weights.get_column(TIME).unique().sort().cast(pl.Int32).to_numpy()
        )
        snapshot_sessions = np.searchsorted(market_ordinals, snapshot_ordinals)
        if not len(snapshot_sessions):
            empty = weights.head(0).select(TIME, ASSET_ID, "weight")
            results[label] = _SparsePortfolioTargets(
                empty,
                empty,
                _empty_unexecuted_weight_keys(),
            )
            continue
        asset_weights = timed_weights.join(market_assets, on=ASSET_ID, how="inner")
        weight_sessions = np.searchsorted(
            market_ordinals,
            asset_weights.get_column(TIME).cast(pl.Int32).to_numpy(),
        )
        weight_assets = (
            asset_weights.get_column(ASSET_ID)
            .cast(asset_enum)
            .to_physical()
            .to_numpy()
        )
        tradable_mask = observed[weight_sessions, weight_assets]
        tradable_actual = asset_weights.filter(pl.Series(tradable_mask))
        actual_sessions = weight_sessions[tradable_mask]
        actual_assets = weight_assets[tradable_mask]

        snapshot_observed = observed[snapshot_sessions]
        weight_snapshot_positions = np.searchsorted(
            snapshot_sessions,
            weight_sessions,
        )
        weight_values = asset_weights.get_column("weight").to_numpy()
        weight_order = np.lexsort((weight_assets, weight_snapshot_positions))
        weight_snapshot_positions = weight_snapshot_positions[weight_order]
        ordered_weight_assets = weight_assets[weight_order]
        ordered_weight_values = weight_values[weight_order]
        retained_weights = np.zeros(market_assets.height, dtype=np.float64)
        desired_weights = np.zeros_like(retained_weights)
        missing_sessions: list[np.ndarray] = []
        missing_assets: list[np.ndarray] = []
        missing_targets: list[np.ndarray] = []
        missing_retained: list[np.ndarray] = []
        weight_offset = 0
        for snapshot_position, snapshot_session in enumerate(snapshot_sessions):
            desired_weights.fill(0.0)
            weight_end = np.searchsorted(
                weight_snapshot_positions,
                snapshot_position,
                side="right",
            )
            selected_assets = ordered_weight_assets[weight_offset:weight_end]
            desired_weights[selected_assets] = ordered_weight_values[
                weight_offset:weight_end
            ]
            absent_changes = np.flatnonzero(
                ~snapshot_observed[snapshot_position]
                & (desired_weights != retained_weights)
            )
            if len(absent_changes):
                missing_sessions.append(
                    np.full(
                        len(absent_changes),
                        snapshot_session,
                        dtype=np.int64,
                    )
                )
                missing_assets.append(absent_changes)
                missing_targets.append(desired_weights[absent_changes].copy())
                missing_retained.append(retained_weights[absent_changes].copy())
            observed_assets = snapshot_observed[snapshot_position]
            retained_weights[observed_assets] = desired_weights[observed_assets]
            weight_offset = weight_end
        if missing_sessions:
            missing_session_indices = np.concatenate(missing_sessions)
            missing_asset_indices = np.concatenate(missing_assets)
            absent_price_targets = pl.DataFrame(
                {
                    TIME: pl.Series(
                        market_ordinals[missing_session_indices],
                        dtype=pl.Int32,
                    ).cast(pl.Date),
                    ASSET_ID: asset_values.gather(
                        pl.Series(missing_asset_indices.astype(np.uint32))
                    ),
                    "target_weight": pl.Series(
                        np.concatenate(missing_targets),
                        dtype=pl.Float64,
                    ),
                    "retained_weight": pl.Series(
                        np.concatenate(missing_retained),
                        dtype=pl.Float64,
                    ),
                }
            ).sort([TIME, ASSET_ID])
        else:
            absent_price_targets = _empty_unexecuted_weight_keys()

        next_positions = np.full(snapshot_observed.shape, -1, dtype=np.int32)
        next_observed = np.full(market_assets.height, -1, dtype=np.int32)
        for position in range(len(snapshot_sessions) - 1, -1, -1):
            next_positions[position] = next_observed
            next_observed[snapshot_observed[position]] = position
        actual_positions = np.searchsorted(snapshot_sessions, actual_sessions)
        exit_positions = next_positions[actual_positions, actual_assets]
        has_exit = exit_positions >= 0
        exit_sessions = snapshot_sessions[exit_positions[has_exit]]
        exit_assets = actual_assets[has_exit]
        if len(exit_sessions):
            exits = pl.DataFrame(
                {
                    TIME: pl.Series(
                        market_ordinals[exit_sessions], dtype=pl.Int32
                    ).cast(pl.Date),
                    ASSET_ID: asset_values.gather(
                        pl.Series(exit_assets.astype(np.uint32))
                    ),
                    "weight": pl.Series(
                        np.zeros(len(exit_sessions)), dtype=pl.Float64
                    ),
                    "_actual": pl.Series(
                        np.zeros(len(exit_sessions), dtype=np.bool_)
                    ),
                }
            )
        else:
            exits = tradable_actual.head(0).select(
                TIME, ASSET_ID, "weight"
            ).with_columns(
                pl.lit(False).alias("_actual"),
            )
        candidate_states = pl.concat(
            [
                exits,
                tradable_actual.select(TIME, ASSET_ID, "weight").with_columns(
                    pl.lit(True).alias("_actual")
                ),
            ]
        )
        target_states = (
            candidate_states.group_by(TIME, ASSET_ID)
            .agg(
                pl.col("weight")
                .filter(pl.col("_actual"))
                .first()
                .fill_null(0.0)
                .alias("weight")
            )
            .sort([ASSET_ID, TIME])
        )
        target_events = (
            target_states.with_columns(
                pl.col("weight")
                .shift(1)
                .over(ASSET_ID)
                .fill_null(0.0)
                .alias("_previous_weight")
            )
            .filter(pl.col("weight") != pl.col("_previous_weight"))
            .select(TIME, ASSET_ID, "weight")
            .sort([TIME, ASSET_ID])
        )

        has_seed = snapshot_observed.any(axis=0)
        seed_positions = snapshot_observed.argmax(axis=0)
        seed_sessions = snapshot_sessions[seed_positions[has_seed]]
        seed_assets = np.flatnonzero(has_seed)
        seed_targets = (
            pl.DataFrame(
                {
                    TIME: pl.Series(
                        market_ordinals[seed_sessions], dtype=pl.Int32
                    ).cast(pl.Date),
                    ASSET_ID: asset_values.gather(
                        pl.Series(seed_assets.astype(np.uint32))
                    ),
                    "weight": pl.Series(
                        np.zeros(len(seed_sessions)), dtype=pl.Float64
                    ),
                }
            )
            .join(
                tradable_actual.select(TIME, ASSET_ID, "weight"),
                on=[TIME, ASSET_ID],
                how="left",
                suffix="_actual",
            )
            .with_columns(
                pl.col("weight_actual").fill_null(pl.col("weight")).alias("weight")
            )
            .select(TIME, ASSET_ID, "weight")
            .sort([TIME, ASSET_ID])
        )
        results[label] = _SparsePortfolioTargets(
            target_events,
            seed_targets,
            absent_price_targets,
        )
    return results


def _resolve_sparse_portfolio_state_from_targets(
    target_events: pl.DataFrame,
    seed_targets: pl.DataFrame,
    execution_availability: pl.DataFrame | None,
    *,
    execution_event_keys: pl.DataFrame,
    retry_blocked: bool,
    first_time: object,
    unexecuted_weight_keys: pl.DataFrame,
) -> _SparsePortfolioState:
    if seed_targets.is_empty():
        raise InputValidationError("at least two overlapping price times are required")
    if target_events.is_empty():
        _, blocks, _ = _apply_execution_availability(
            target_events,
            None,
        )
        return _SparsePortfolioState(
            target_events=seed_targets.sort([TIME, ASSET_ID]),
            executable_events=seed_targets.with_columns(
                pl.lit(0.0).alias("weight")
            ).sort([TIME, ASSET_ID]),
            execution_blocks=blocks,
            execution_event_count=0,
            first_time=first_time,
            unexecuted_weight_keys=unexecuted_weight_keys,
        )
    sparse_desired = _sparse_execution_desired_weights(
        target_events,
        execution_event_keys,
    )
    executable_events, blocks, event_count = _apply_execution_availability(
        sparse_desired,
        execution_availability,
        retry_blocked=retry_blocked,
        availability_validated=True,
        target_events=target_events,
    )
    target_frame_events = (
        pl.concat([seed_targets, target_events])
        .group_by(TIME, ASSET_ID)
        .last()
        .sort([TIME, ASSET_ID])
    )
    executable_frame_events = (
        pl.concat(
            [
                seed_targets.with_columns(pl.lit(0.0).alias("weight")),
                executable_events,
            ]
        )
        .group_by(TIME, ASSET_ID)
        .last()
        .sort([TIME, ASSET_ID])
    )
    return _SparsePortfolioState(
        target_events=target_frame_events,
        executable_events=executable_frame_events,
        execution_blocks=blocks,
        execution_event_count=event_count,
        first_time=first_time,
        unexecuted_weight_keys=unexecuted_weight_keys,
    )


def _calculate_sparse_portfolio_batch(
    states: Mapping[str, _SparsePortfolioState],
    market_context: _SparseMarketContext,
    *,
    config: BacktestConfig,
    slippage_rates: pl.DataFrame | None,
) -> dict[str, _CompactBacktestResult]:
    labels = list(states)
    portfolio_count = len(labels)
    session_lookup = market_context.return_sessions
    asset_lookup = market_context.return_assets
    return_matrix = market_context.return_matrix

    labeled_events = []
    for portfolio_index, label in enumerate(labels):
        labeled_events.append(
            states[label].executable_events.with_columns(
                pl.lit(portfolio_index, dtype=pl.UInt32).alias("_portfolio_index")
            )
        )
    events = (
        pl.concat(labeled_events)
        .join(session_lookup, on=TIME, how="inner")
        .join(asset_lookup, on=ASSET_ID, how="inner")
        .sort(["_session_index", "_portfolio_index", "_asset_index"])
    )
    events = _attach_slippage_rates(
        events,
        slippage_rates,
        default_rate=config.transaction_cost.slippage_rate,
    )
    event_sessions = events.get_column("_session_index").to_numpy()
    event_portfolios = events.get_column("_portfolio_index").to_numpy()
    event_assets = events.get_column("_asset_index").to_numpy()
    event_weights = events.get_column("weight").to_numpy()
    event_slippage_rates = events.get_column("_slippage_rate").to_numpy()
    event_slippage_fallbacks = events.get_column(
        "_slippage_fallback"
    ).to_numpy()

    periods = session_lookup.height
    holdings = np.zeros((portfolio_count, asset_lookup.height), dtype=np.float64)
    gross_returns = np.zeros((periods, portfolio_count), dtype=np.float64)
    net_returns = np.zeros_like(gross_returns)
    turnover_values = np.zeros_like(gross_returns)
    traded_asset_counts = np.zeros((periods, portfolio_count), dtype=np.int64)
    slippage_fallback_asset_counts = np.zeros_like(traded_asset_counts)
    traded_notionals = np.zeros_like(gross_returns)
    buy_notionals = np.zeros_like(gross_returns)
    sell_notionals = np.zeros_like(gross_returns)
    slippage_fees = np.zeros_like(gross_returns)
    raw_fees = np.zeros_like(gross_returns)
    min_fee_adjustments = np.zeros_like(gross_returns)
    commission_fees = np.zeros_like(gross_returns)
    stamp_tax_fees = np.zeros_like(gross_returns)
    total_fees = np.zeros_like(gross_returns)
    requested_total_fees = np.zeros_like(gross_returns)
    unfunded_fees = np.zeros_like(gross_returns)
    cost_returns = np.zeros_like(gross_returns)
    is_bankrupt = np.zeros((periods, portfolio_count), dtype=np.bool_)
    bankruptcy_events = np.zeros_like(is_bankrupt)
    gross_value_path = np.zeros_like(gross_returns)
    net_value_path = np.zeros_like(gross_returns)
    current_gross = np.full(portfolio_count, config.initial_capital, dtype=np.float64)
    current_net = np.full(portfolio_count, config.initial_capital, dtype=np.float64)
    bankrupt = np.zeros(portfolio_count, dtype=np.bool_)

    first_session_indices = np.array(
        [
            session_lookup.get_column(TIME).search_sorted(
                states[label].first_time
            )
            for label in labels
        ],
        dtype=np.int64,
    )
    event_offset = 0
    for session_index in range(periods):
        changed_portfolios: list[int] = []
        changed_deltas: list[float] = []
        changed_slippage_rates: list[float] = []
        changed_slippage_fallbacks: list[bool] = []
        while (
            event_offset < events.height
            and int(event_sessions[event_offset]) == session_index
        ):
            portfolio_index = int(event_portfolios[event_offset])
            asset_index = int(event_assets[event_offset])
            target = float(event_weights[event_offset])
            if bankrupt[portfolio_index]:
                event_offset += 1
                continue
            delta = target - holdings[portfolio_index, asset_index]
            holdings[portfolio_index, asset_index] = target
            if delta != 0.0:
                changed_portfolios.append(portfolio_index)
                changed_deltas.append(delta)
                changed_slippage_rates.append(
                    float(event_slippage_rates[event_offset])
                )
                changed_slippage_fallbacks.append(
                    bool(event_slippage_fallbacks[event_offset])
                )
            event_offset += 1

        if changed_portfolios:
            changed_portfolio_array = np.asarray(
                changed_portfolios, dtype=np.int64
            )
            changed_delta_array = np.asarray(changed_deltas, dtype=np.float64)
            absolute_delta_array = np.abs(changed_delta_array)
            changed_slippage_rate_array = np.asarray(
                changed_slippage_rates,
                dtype=np.float64,
            )
            changed_slippage_fallback_array = np.asarray(
                changed_slippage_fallbacks,
                dtype=np.int64,
            )
            np.add.at(
                turnover_values[session_index],
                changed_portfolio_array,
                absolute_delta_array,
            )
            np.add.at(
                traded_asset_counts[session_index],
                changed_portfolio_array,
                1,
            )
            np.add.at(
                slippage_fallback_asset_counts[session_index],
                changed_portfolio_array,
                changed_slippage_fallback_array,
            )
            per_trade_notional = (
                absolute_delta_array * current_net[changed_portfolio_array]
            )
            per_trade_buy_notional = (
                np.maximum(changed_delta_array, 0.0)
                * current_net[changed_portfolio_array]
            )
            per_trade_sell_notional = (
                np.maximum(-changed_delta_array, 0.0)
                * current_net[changed_portfolio_array]
            )
            per_trade_raw = (
                per_trade_notional * config.transaction_cost.rate
            )
            per_trade_commission = np.maximum(
                per_trade_raw,
                config.transaction_cost.min_fee,
            )
            per_trade_slippage = (
                per_trade_notional * changed_slippage_rate_array
            )
            per_trade_stamp_tax = (
                per_trade_sell_notional
                * config.transaction_cost.stamp_tax_rate
            )
            for target, values in (
                (buy_notionals, per_trade_buy_notional),
                (sell_notionals, per_trade_sell_notional),
                (slippage_fees, per_trade_slippage),
                (commission_fees, per_trade_commission),
                (stamp_tax_fees, per_trade_stamp_tax),
            ):
                np.add.at(
                    target[session_index],
                    changed_portfolio_array,
                    values,
                )
            np.add.at(
                raw_fees[session_index],
                changed_portfolio_array,
                per_trade_raw,
            )

        active = session_index >= first_session_indices
        gross_returns[session_index] = holdings @ return_matrix[session_index]
        gross_returns[session_index, bankrupt] = 0.0
        traded_notionals[session_index] = (
            turnover_values[session_index] * current_net
        )
        min_fee_adjustments[session_index] = (
            commission_fees[session_index] - raw_fees[session_index]
        )
        total_fees[session_index] = (
            commission_fees[session_index]
            + slippage_fees[session_index]
            + stamp_tax_fees[session_index]
        )
        requested_total_fees[session_index] = total_fees[session_index]
        np.divide(
            total_fees[session_index],
            current_net,
            out=cost_returns[session_index],
            where=current_net != 0.0,
        )
        net_returns[session_index] = (
            gross_returns[session_index] - cost_returns[session_index]
        )
        next_net = current_net * (1.0 + net_returns[session_index])
        failed_mask = active & ~bankrupt & (next_net <= 0.0)
        if np.any(failed_mask) and config.insolvency_action == "raise":
            failed = int(np.flatnonzero(failed_mask)[0])
            time = session_lookup.get_column(TIME)[session_index]
            raise InputValidationError(
                "net portfolio value became non-positive after transaction costs "
                f"at {time}: current_value={current_net[failed]:.6g}, "
                f"gross_return={gross_returns[session_index, failed]:.6g}, "
                f"cost_return={cost_returns[session_index, failed]:.6g}, "
                "Increase initial_capital or reduce traded universe/turnover."
            )
        if np.any(failed_mask):
            failed_indices = np.flatnonzero(failed_mask)
            available_wealth = np.maximum(
                current_net[failed_indices]
                * (1.0 + gross_returns[session_index, failed_indices]),
                0.0,
            )
            requested_fees = requested_total_fees[
                session_index, failed_indices
            ]
            scales = np.divide(
                available_wealth,
                requested_fees,
                out=np.zeros_like(available_wealth),
                where=requested_fees > 0.0,
            )
            scales = np.minimum(scales, 1.0)
            for fee_path in (
                slippage_fees,
                raw_fees,
                min_fee_adjustments,
                commission_fees,
                stamp_tax_fees,
            ):
                fee_path[session_index, failed_indices] *= scales
            total_fees[session_index, failed_indices] = available_wealth
            unfunded_fees[session_index, failed_indices] = np.maximum(
                requested_fees - available_wealth,
                0.0,
            )
            cost_returns[session_index, failed_indices] = np.divide(
                available_wealth,
                current_net[failed_indices],
                out=np.zeros_like(available_wealth),
                where=current_net[failed_indices] != 0.0,
            )
            net_returns[session_index, failed_indices] = -1.0
            next_net[failed_indices] = 0.0
            bankruptcy_events[session_index, failed_indices] = True
            bankrupt[failed_indices] = True
        is_bankrupt[session_index] = bankrupt
        current_gross *= 1.0 + gross_returns[session_index]
        current_net = next_net
        gross_value_path[session_index] = current_gross
        net_value_path[session_index] = current_net

    times = session_lookup.get_column(TIME)
    results: dict[str, _CompactBacktestResult] = {}
    for portfolio_index, label in enumerate(labels):
        start = first_session_indices[portfolio_index]
        selected_times = times.slice(int(start))
        gross = gross_returns[start:, portfolio_index]
        net = net_returns[start:, portfolio_index]
        returns = pl.DataFrame(
            {
                TIME: selected_times,
                "gross_return": gross,
                "net_return": net,
                "is_bankrupt": is_bankrupt[start:, portfolio_index],
                "bankruptcy_event": bankruptcy_events[start:, portfolio_index],
            }
        )
        turn = pl.DataFrame(
            {TIME: selected_times, "turnover": turnover_values[start:, portfolio_index]}
        )
        costs_data = pl.DataFrame(
            {
                TIME: selected_times,
                "traded_asset_count": traded_asset_counts[start:, portfolio_index],
                "slippage_fallback_asset_count": slippage_fallback_asset_counts[
                    start:, portfolio_index
                ],
                "traded_notional": traded_notionals[start:, portfolio_index],
                "buy_notional": buy_notionals[start:, portfolio_index],
                "sell_notional": sell_notionals[start:, portfolio_index],
                "slippage_fee": slippage_fees[start:, portfolio_index],
                "raw_fee": raw_fees[start:, portfolio_index],
                "min_fee_adjustment": min_fee_adjustments[start:, portfolio_index],
                "commission_fee": commission_fees[start:, portfolio_index],
                "stamp_tax_fee": stamp_tax_fees[start:, portfolio_index],
                "total_fee": total_fees[start:, portfolio_index],
                "requested_total_fee": requested_total_fees[
                    start:, portfolio_index
                ],
                "unfunded_fee": unfunded_fees[start:, portfolio_index],
                "cost_return": cost_returns[start:, portfolio_index],
            }
        )
        value = pl.DataFrame(
            {
                TIME: selected_times,
                "gross_value": gross_value_path[start:, portfolio_index],
                "net_value": net_value_path[start:, portfolio_index],
                "gross_return_cumulative": np.cumprod(1.0 + gross) - 1.0,
                "net_return_cumulative": np.cumprod(1.0 + net) - 1.0,
            }
        )
        costs = TransactionCostBreakdown(costs_data)
        result_state = states[label]
        bankruptcy_positions = np.flatnonzero(
            bankruptcy_events[start:, portfolio_index]
        )
        if bankruptcy_positions.size:
            bankruptcy_time = selected_times[int(bankruptcy_positions[0])]
            result_state = replace(
                result_state,
                executable_events=result_state.executable_events.filter(
                    pl.col(TIME) <= bankruptcy_time
                ),
                execution_blocks=result_state.execution_blocks.filter(
                    pl.col(TIME) <= bankruptcy_time
                ),
            )
        summary, performance = summarize_performance(
            returns=returns,
            turnover=turn,
            costs=costs,
            initial_capital=config.initial_capital,
            annualization=config.annualization,
        )
        results[label] = _CompactBacktestResult(
            returns=returns,
            value=value,
            turnover=turn,
            costs=costs,
            summary=summary,
            performance=performance,
            state=result_state,
        )
    return results


def _execution_rule_event_keys(
    execution_availability: pl.DataFrame | None,
    session_ordinals: np.ndarray,
    asset_values: pl.Series,
    present: np.ndarray,
) -> pl.DataFrame:
    if execution_availability is None or execution_availability.is_empty():
        return pl.DataFrame(schema={TIME: pl.Date, ASSET_ID: pl.String})
    asset_enum = pl.Enum(asset_values.to_list())
    rules = execution_availability.select(TIME, ASSET_ID).join(
        asset_values.to_frame(),
        on=ASSET_ID,
        how="inner",
    )
    rule_assets = (
        rules.get_column(ASSET_ID)
        .cast(asset_enum)
        .to_physical()
        .to_numpy()
    )
    rule_ordinals = rules.get_column(TIME).cast(pl.Int32).to_numpy()
    rule_sessions = np.searchsorted(session_ordinals, rule_ordinals)
    event_ordinals: list[int] = []
    event_assets: list[int] = []
    for rule_ordinal, session_index, asset_index in zip(
        rule_ordinals,
        rule_sessions,
        rule_assets,
        strict=True,
    ):
        if (
            session_index >= len(session_ordinals)
            or session_ordinals[session_index] != rule_ordinal
            or not present[session_index, asset_index]
        ):
            continue
        event_ordinals.append(int(rule_ordinal))
        event_assets.append(int(asset_index))
        next_session = int(session_index) + 1
        while (
            next_session < len(session_ordinals)
            and not present[next_session, asset_index]
        ):
            next_session += 1
        if next_session < len(session_ordinals):
            event_ordinals.append(int(session_ordinals[next_session]))
            event_assets.append(int(asset_index))
    return (
        pl.DataFrame(
            {
                TIME: pl.Series(event_ordinals, dtype=pl.Int32).cast(pl.Date),
                ASSET_ID: asset_values.gather(
                    pl.Series(event_assets, dtype=pl.UInt32)
                ),
            }
        )
        .unique()
        .sort([TIME, ASSET_ID])
    )


def _sparse_execution_desired_weights(
    target_events: pl.DataFrame,
    execution_event_keys: pl.DataFrame,
) -> pl.DataFrame:
    if execution_event_keys.is_empty():
        return target_events
    active_assets = target_events.select(ASSET_ID).unique()
    active_event_keys = execution_event_keys.join(
        active_assets,
        on=ASSET_ID,
        how="inner",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        return (
            pl.concat([target_events.select(TIME, ASSET_ID), active_event_keys])
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
        turnover=core.turnover,
        costs=core.costs,
        summary=core.summary,
        performance=core.performance,
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
    initial_is_bankrupt: bool = False,
    execution_availability_validated: bool = False,
    prepared_execution_keys: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
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
        slippage_rates=slippage_rates,
        initial_gross_value=initial_gross_value,
        initial_net_value=initial_net_value,
        initial_is_bankrupt=initial_is_bankrupt,
    )
    bankruptcy_time = (
        returns.filter(pl.col("bankruptcy_event")).get_column(TIME).min()
    )
    if initial_is_bankrupt and initial_executed_weights is not None:
        first_return_time = returns.get_column(TIME).min()
        executable_weights = initial_executed_weights.select(
            pl.lit(first_return_time, dtype=pl.Date).alias(TIME),
            ASSET_ID,
            "weight",
        )
        execution_blocks = execution_blocks.head(0)
    elif bankruptcy_time is not None:
        executable_weights = executable_weights.filter(
            pl.col(TIME) <= bankruptcy_time
        )
        execution_blocks = execution_blocks.filter(pl.col(TIME) <= bankruptcy_time)
    turn = turn.join(
        returns.select(TIME, "is_bankrupt", "bankruptcy_event"),
        on=TIME,
        how="left",
    ).with_columns(
        pl.when(pl.col("is_bankrupt") & ~pl.col("bankruptcy_event"))
        .then(0.0)
        .otherwise(pl.col("turnover"))
        .alias("turnover")
    ).drop("is_bankrupt", "bankruptcy_event")
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
    slippage_rates: pl.DataFrame | None,
    initial_gross_value: float | None = None,
    initial_net_value: float | None = None,
    initial_is_bankrupt: bool = False,
) -> tuple[TransactionCostBreakdown, pl.DataFrame, pl.DataFrame]:
    trade_summary = _trade_summary(
        deltas,
        slippage_rates=slippage_rates,
        default_slippage_rate=config.transaction_cost.slippage_rate,
    )
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
    requested_gross_values = timeline.get_column("gross_return").to_numpy()
    gross_values = requested_gross_values.copy()
    delta_lists = timeline.get_column("signed_weight_deltas")
    slippage_rate_lists = timeline.get_column("slippage_rates")
    slippage_fallback_lists = timeline.get_column("slippage_fallbacks")
    traded_asset_counts = (
        delta_lists.list.len().fill_null(0).to_numpy().astype(np.int64)
    )
    flat_deltas = (
        delta_lists.explode(empty_as_null=True).drop_nulls().to_numpy()
    )
    flat_slippage_rates = (
        slippage_rate_lists.explode(empty_as_null=True).drop_nulls().to_numpy()
    )
    flat_slippage_fallbacks = (
        slippage_fallback_lists.explode(empty_as_null=True)
        .drop_nulls()
        .to_numpy()
    )
    traded_notionals = np.empty(periods, dtype=np.float64)
    buy_notionals = np.empty(periods, dtype=np.float64)
    sell_notionals = np.empty(periods, dtype=np.float64)
    slippage_fees = np.empty(periods, dtype=np.float64)
    slippage_fallback_asset_counts = np.empty(periods, dtype=np.int64)
    raw_fees = np.empty(periods, dtype=np.float64)
    min_fee_adjustments = np.empty(periods, dtype=np.float64)
    commission_fees = np.empty(periods, dtype=np.float64)
    stamp_tax_fees = np.empty(periods, dtype=np.float64)
    total_fees = np.empty(periods, dtype=np.float64)
    requested_total_fees = np.empty(periods, dtype=np.float64)
    unfunded_fees = np.empty(periods, dtype=np.float64)
    cost_returns = np.empty(periods, dtype=np.float64)
    net_returns = np.empty(periods, dtype=np.float64)
    gross_value_path = np.empty(periods, dtype=np.float64)
    net_value_path = np.empty(periods, dtype=np.float64)
    is_bankrupt = np.empty(periods, dtype=np.bool_)
    bankruptcy_events = np.zeros(periods, dtype=np.bool_)

    offset = 0
    bankrupt = initial_is_bankrupt
    for position in range(periods):
        time = times[position]
        traded_asset_count = int(traded_asset_counts[position])
        signed_weight_deltas = flat_deltas[offset : offset + traded_asset_count]
        trade_slippage_rates = flat_slippage_rates[
            offset : offset + traded_asset_count
        ]
        trade_slippage_fallbacks = flat_slippage_fallbacks[
            offset : offset + traded_asset_count
        ]
        offset += traded_asset_count
        if bankrupt:
            gross_values[position] = 0.0
            traded_asset_counts[position] = 0
            traded_notionals[position] = 0.0
            buy_notionals[position] = 0.0
            sell_notionals[position] = 0.0
            slippage_fees[position] = 0.0
            slippage_fallback_asset_counts[position] = 0
            raw_fees[position] = 0.0
            min_fee_adjustments[position] = 0.0
            commission_fees[position] = 0.0
            stamp_tax_fees[position] = 0.0
            total_fees[position] = 0.0
            requested_total_fees[position] = 0.0
            unfunded_fees[position] = 0.0
            cost_returns[position] = 0.0
            net_returns[position] = 0.0
            gross_value_path[position] = current_gross_value
            net_value_path[position] = current_net_value
            is_bankrupt[position] = True
            continue
        weight_deltas = np.abs(signed_weight_deltas)
        weight_delta = float(np.sum(weight_deltas))
        traded_notional = weight_delta * current_net_value
        buy_notional = float(
            np.maximum(signed_weight_deltas, 0.0).sum() * current_net_value
        )
        sell_notional = float(
            np.maximum(-signed_weight_deltas, 0.0).sum() * current_net_value
        )
        slippage_fee = float(
            (weight_deltas * current_net_value * trade_slippage_rates).sum()
        )
        raw_fee = traded_notional * config.transaction_cost.rate
        commission_fee = float(
            np.maximum(
                weight_deltas * current_net_value * config.transaction_cost.rate,
                config.transaction_cost.min_fee,
            ).sum()
        )
        stamp_tax_fee = sell_notional * config.transaction_cost.stamp_tax_rate
        total_fee = commission_fee + slippage_fee + stamp_tax_fee

        cost_return = total_fee / current_net_value if current_net_value else 0.0
        gross_return = float(requested_gross_values[position])
        net_return = gross_return - cost_return
        next_net_value = current_net_value * (1.0 + net_return)
        if next_net_value <= 0.0 and config.insolvency_action == "raise":
            raise InputValidationError(
                "net portfolio value became non-positive after transaction costs "
                f"at {time}: current_value={current_net_value:.6g}, "
                f"gross_return={gross_return:.6g}, "
                f"cost_return={cost_return:.6g}, "
                f"traded_asset_count={traded_asset_count}, "
                f"total_fee={total_fee:.6g}. "
                "Increase initial_capital or reduce traded universe/turnover."
            )
        requested_total_fee = total_fee
        unfunded_fee = 0.0
        if next_net_value <= 0.0:
            available_wealth = max(
                current_net_value * (1.0 + gross_return),
                0.0,
            )
            scale = (
                min(available_wealth / requested_total_fee, 1.0)
                if requested_total_fee > 0.0
                else 0.0
            )
            slippage_fee *= scale
            raw_fee *= scale
            commission_fee *= scale
            stamp_tax_fee *= scale
            total_fee = available_wealth
            unfunded_fee = max(requested_total_fee - total_fee, 0.0)
            cost_return = (
                total_fee / current_net_value if current_net_value else 0.0
            )
            net_return = -1.0
            next_net_value = 0.0
            bankruptcy_events[position] = True
            bankrupt = True
        current_gross_value *= 1.0 + gross_return
        current_net_value = next_net_value
        traded_notionals[position] = traded_notional
        buy_notionals[position] = buy_notional
        sell_notionals[position] = sell_notional
        slippage_fees[position] = slippage_fee
        slippage_fallback_asset_counts[position] = int(
            np.sum(trade_slippage_fallbacks)
        )
        raw_fees[position] = raw_fee
        min_fee_adjustments[position] = commission_fee - raw_fee
        commission_fees[position] = commission_fee
        stamp_tax_fees[position] = stamp_tax_fee
        total_fees[position] = total_fee
        requested_total_fees[position] = requested_total_fee
        unfunded_fees[position] = unfunded_fee
        cost_returns[position] = cost_return
        net_returns[position] = net_return
        gross_value_path[position] = current_gross_value
        net_value_path[position] = current_net_value
        is_bankrupt[position] = bankrupt

    costs = pl.DataFrame(
        {
            TIME: times,
            "traded_asset_count": traded_asset_counts,
            "slippage_fallback_asset_count": slippage_fallback_asset_counts,
            "traded_notional": traded_notionals,
            "buy_notional": buy_notionals,
            "sell_notional": sell_notionals,
            "slippage_fee": slippage_fees,
            "raw_fee": raw_fees,
            "min_fee_adjustment": min_fee_adjustments,
            "commission_fee": commission_fees,
            "stamp_tax_fee": stamp_tax_fees,
            "total_fee": total_fees,
            "requested_total_fee": requested_total_fees,
            "unfunded_fee": unfunded_fees,
            "cost_return": cost_returns,
        }
    )
    returns = pl.DataFrame(
        {
            TIME: times,
            "gross_return": gross_values,
            "net_return": net_returns,
            "is_bankrupt": is_bankrupt,
            "bankruptcy_event": bankruptcy_events,
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
    *,
    slippage_rates: pl.DataFrame | None,
    default_slippage_rate: float,
) -> pl.DataFrame:
    trades = _attach_slippage_rates(
        deltas,
        slippage_rates,
        default_rate=default_slippage_rate,
    )
    return (
        trades.group_by(TIME)
        .agg(
            pl.col("signed_weight_delta")
            .filter(pl.col("weight_delta") > 0.0)
            .alias("signed_weight_deltas"),
            pl.col("_slippage_rate")
            .filter(pl.col("weight_delta") > 0.0)
            .alias("slippage_rates"),
            pl.col("_slippage_fallback")
            .filter(pl.col("weight_delta") > 0.0)
            .alias("slippage_fallbacks"),
        )
        .sort(TIME)
    )


def _attach_slippage_rates(
    trades: pl.DataFrame,
    slippage_rates: pl.DataFrame | None,
    *,
    default_rate: float,
) -> pl.DataFrame:
    """Attach the latest effective per-asset rate to sparse trade events."""

    indexed = trades.with_row_index("_trade_order")
    if slippage_rates is None:
        return indexed.with_columns(
            pl.lit(default_rate).alias("_slippage_rate"),
            pl.lit(False).alias("_slippage_fallback"),
        ).drop("_trade_order")
    schedule = slippage_rates
    if "is_fallback" not in schedule.columns:
        schedule = schedule.with_columns(pl.lit(False).alias("is_fallback"))
    joined = (
        indexed.sort([ASSET_ID, TIME])
        .join_asof(
            schedule.sort([ASSET_ID, TIME]),
            on=TIME,
            by=ASSET_ID,
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            pl.col("slippage_rate")
            .fill_null(default_rate)
            .alias("_slippage_rate"),
            (
                pl.col("slippage_rate").is_null()
                | pl.col("is_fallback").fill_null(False)
            ).alias("_slippage_fallback"),
        )
        .sort("_trade_order")
    )
    return joined.drop(
        "_trade_order",
        "slippage_rate",
        "is_fallback",
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
