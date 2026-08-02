"""Persistent portfolio paths and date-window result sections."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from .benchmarks import benchmark_performance, top_n_excess_returns
from .config import BacktestConfig
from .engine import (
    _backtest_weight_frame_with_forward_returns,
    backtest_weight_frame,
)
from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME, validate_prices, validate_weights
from .performance import summarize_performance
from .results import BacktestResult, PerformanceSummary, TransactionCostBreakdown
from .returns import _prepare_price_data, cumulative_returns

PORTFOLIO_PATH_VERSION = 2
RESULT_SECTION_VERSION = 2
RESULT_SECTIONS = (
    "Summary & Coverage",
    "IC & ICIR",
    "Quantiles",
    "TOP N",
    "TOP N vs Benchmarks",
    "Spread Performance",
)


@dataclass(frozen=True, slots=True)
class PortfolioPathIdentity:
    """Immutable inputs that distinguish one reusable portfolio path."""

    alpha_revision: str
    universe: str
    policy_combo: str
    parameters_hash: str = ""
    market_data_hash: str = ""
    engine_version: int = PORTFOLIO_PATH_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("alpha_revision", self.alpha_revision),
            ("universe", self.universe),
            ("policy_combo", self.policy_combo),
        ):
            if not value.strip():
                raise ValueError(f"portfolio path {label} must not be blank")

    @property
    def content_hash(self) -> str:
        payload = {
            "alpha_revision": self.alpha_revision,
            "universe": self.universe,
            "policy_combo": self.policy_combo,
            "parameters_hash": self.parameters_hash,
            "market_data_hash": self.market_data_hash,
            "engine_version": self.engine_version,
        }
        return hashlib.blake2b(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            digest_size=16,
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class PortfolioStateCheckpoint:
    """State required to continue a path without replaying prior sessions."""

    time: date
    target_weights: pl.DataFrame
    executed_weights: pl.DataFrame
    gross_value: float
    net_value: float


@dataclass(frozen=True, slots=True)
class PortfolioPathChunk:
    """One appendable interval of a continuous portfolio path."""

    identity: PortfolioPathIdentity
    returns: pl.DataFrame
    turnover: pl.DataFrame
    costs: pl.DataFrame
    target_weight_changes: pl.DataFrame
    executed_weight_changes: pl.DataFrame
    execution_blocks: pl.DataFrame
    checkpoint: PortfolioStateCheckpoint
    series: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    execution_event_count: int = 0


@dataclass(frozen=True, slots=True)
class ResultSectionSpec:
    """One ordered Results section and its selected item keys."""

    section: str
    items: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.section not in RESULT_SECTIONS:
            raise ValueError(f"unknown result section: {self.section}")
        if len(set(self.items)) != len(self.items):
            raise ValueError("result section item keys must be unique")


@dataclass(frozen=True, slots=True)
class ResultWindow:
    """Inclusive observation bounds for complete return intervals."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("result window start must not follow end")


@dataclass(frozen=True, slots=True)
class ResultSection:
    """A section result that can be published independently."""

    spec: ResultSectionSpec
    window: ResultWindow
    metrics: Mapping[str, float | int | None]
    tables: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    version: int = RESULT_SECTION_VERSION


def materialize_portfolio_path(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    identity: PortfolioPathIdentity,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None = None,
    prepared_forward_returns: pl.DataFrame | None = None,
    prepared_price_gaps: pl.DataFrame | None = None,
    prepared_active_assets: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
) -> PortfolioPathChunk:
    """Build the first chunk of a reusable continuous portfolio path."""

    aligned_weights = validate_weights(weights)
    aligned_prices = validate_prices(prices)
    if prepared_forward_returns is None:
        result = backtest_weight_frame(
            aligned_weights,
            aligned_prices,
            config=config,
            execution_availability=execution_availability,
            slippage_rates=slippage_rates,
        )
    else:
        result = _backtest_weight_frame_with_forward_returns(
            aligned_weights,
            aligned_prices,
            prepared_forward_returns,
            config=config,
            price_gaps=prepared_price_gaps,
            execution_availability=execution_availability,
            prepared_active_assets=prepared_active_assets,
            slippage_rates=slippage_rates,
        )
    return _path_chunk(
        identity,
        result,
        aligned_prices,
        initial_target_weights=None,
        initial_executed_weights=None,
    )


def portfolio_path_from_backtest(
    result: BacktestResult,
    prices: pl.DataFrame,
    *,
    identity: PortfolioPathIdentity,
    series: Mapping[str, pl.DataFrame] | None = None,
) -> PortfolioPathChunk:
    """Publish an already-computed compatible backtest without rerunning it."""

    chunk = _path_chunk(
        identity,
        result,
        validate_prices(prices),
        initial_target_weights=None,
        initial_executed_weights=None,
    )
    if not series:
        return chunk
    return PortfolioPathChunk(
        identity=chunk.identity,
        returns=chunk.returns,
        turnover=chunk.turnover,
        costs=chunk.costs,
        target_weight_changes=chunk.target_weight_changes,
        executed_weight_changes=chunk.executed_weight_changes,
        execution_blocks=chunk.execution_blocks,
        checkpoint=chunk.checkpoint,
        series=dict(series),
        execution_event_count=chunk.execution_event_count,
    )


def resume_portfolio_path(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    identity: PortfolioPathIdentity,
    checkpoint: PortfolioStateCheckpoint,
    config: BacktestConfig,
    execution_availability: pl.DataFrame | None = None,
    prepared_forward_returns: pl.DataFrame | None = None,
    prepared_price_gaps: pl.DataFrame | None = None,
    prepared_active_assets: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
) -> PortfolioPathChunk:
    """Continue a path from a checkpoint without charging a new initial trade."""

    aligned_weights = validate_weights(weights)
    aligned_prices = validate_prices(prices)
    price_data = (
        _prepare_price_data(aligned_prices, inputs_sorted=True)
        if prepared_forward_returns is None
        else None
    )
    forward_returns = (
        price_data.forward_returns
        if price_data is not None
        else prepared_forward_returns
    )
    price_gaps = (
        price_data.price_gaps
        if price_data is not None
        else prepared_price_gaps
    )
    if forward_returns.is_empty():
        raise InputValidationError("resume prices require at least one return interval")
    first_time = forward_returns.get_column(TIME).min()
    if first_time != checkpoint.time:
        raise InputValidationError(
            "resume prices must begin at the checkpoint time"
        )
    carry = checkpoint.target_weights.select(
        pl.lit(first_time, dtype=pl.Date).alias(TIME),
        ASSET_ID,
        "weight",
    )
    continued_weights = (
        aligned_weights
        if aligned_weights.filter(pl.col(TIME) == first_time).height
        else pl.concat([carry, aligned_weights])
    )
    continued_weights = continued_weights.sort([TIME, ASSET_ID])
    result = _backtest_weight_frame_with_forward_returns(
        continued_weights,
        aligned_prices,
        forward_returns,
        config=config,
        price_gaps=price_gaps,
        execution_availability=execution_availability,
        initial_target_weights=checkpoint.target_weights,
        initial_executed_weights=checkpoint.executed_weights,
        initial_gross_value=checkpoint.gross_value,
        initial_net_value=checkpoint.net_value,
        prepared_active_assets=prepared_active_assets,
        slippage_rates=slippage_rates,
    )
    return _path_chunk(
        identity,
        result,
        aligned_prices,
        initial_target_weights=checkpoint.target_weights,
        initial_executed_weights=checkpoint.executed_weights,
    )


def compute_result_section(
    path: PortfolioPathChunk,
    spec: ResultSectionSpec,
    window: ResultWindow,
    *,
    annualization: int,
) -> ResultSection:
    """Compute one independently publishable section from a continuous path."""

    intervals = path.returns.filter(
        (pl.col(TIME) >= window.start)
        & (pl.col("next_time") <= window.end)
    ).sort(TIME)
    if intervals.is_empty():
        raise InputValidationError(
            "result window contains no complete portfolio return intervals"
        )
    turnover = path.turnover.join(
        intervals.select(TIME),
        on=TIME,
        how="inner",
    )
    costs = path.costs.join(intervals.select(TIME), on=TIME, how="inner")
    returns = intervals.select(TIME, "gross_return", "net_return")
    summary, performance = summarize_performance(
        returns=returns,
        turnover=turnover,
        costs=TransactionCostBreakdown(costs),
        initial_capital=1.0,
        annualization=annualization,
    )
    value = (
        cumulative_returns(returns, "gross_return")
        .join(cumulative_returns(returns, "net_return"), on=TIME)
        .sort(TIME)
    )
    metrics: dict[str, float | int | None] = (
        _summary_metrics(summary)
        if spec.section
        in {"Summary & Coverage", "TOP N", "TOP N vs Benchmarks"}
        else {}
    )
    tables = {
        "returns": intervals,
        "value": value,
        "performance": performance,
        "turnover": turnover,
        "costs": costs,
    }
    _attach_section_series(
        path,
        spec,
        window,
        metrics,
        tables,
        annualization=annualization,
    )
    return ResultSection(
        spec=spec,
        window=window,
        metrics=metrics,
        tables=tables,
    )


def _path_chunk(
    identity: PortfolioPathIdentity,
    result,
    prices: pl.DataFrame,
    *,
    initial_target_weights: pl.DataFrame | None,
    initial_executed_weights: pl.DataFrame | None,
) -> PortfolioPathChunk:
    sessions = (
        prices.select(TIME)
        .unique()
        .sort(TIME)
        .with_columns(pl.col(TIME).shift(-1).alias("next_time"))
        .drop_nulls("next_time")
    )
    returns = result.returns.join(sessions, on=TIME, how="inner").select(
        TIME,
        "next_time",
        "gross_return",
        "net_return",
    )
    if returns.is_empty():
        raise InputValidationError("portfolio path contains no complete intervals")
    last_time = result.returns.get_column(TIME).max()
    checkpoint_time = returns.get_column("next_time").max()
    target_weights = _latest_checkpoint_weights(
        result.target_weights,
        at_or_before=last_time,
    )
    executed_weights = _latest_checkpoint_weights(
        result.weights,
        at_or_before=last_time,
    )
    checkpoint = PortfolioStateCheckpoint(
        time=checkpoint_time,
        target_weights=target_weights,
        executed_weights=executed_weights,
        gross_value=float(result.value.get_column("gross_value")[-1]),
        net_value=float(result.value.get_column("net_value")[-1]),
    )
    target_changes = _sparse_weight_changes(
        result.target_weights,
        initial_target_weights,
    )
    executed_changes = _sparse_weight_changes(
        result.weights,
        initial_executed_weights,
    )
    event_count = (
        pl.concat(
            [
                target_changes.select(TIME, ASSET_ID),
                executed_changes.select(TIME, ASSET_ID),
                result.execution_blocks.select(TIME, ASSET_ID),
            ]
        )
        .unique()
        .height
    )
    return PortfolioPathChunk(
        identity=identity,
        returns=returns,
        turnover=result.turnover,
        costs=result.transaction_costs.data,
        target_weight_changes=target_changes,
        executed_weight_changes=executed_changes,
        execution_blocks=result.execution_blocks,
        checkpoint=checkpoint,
        execution_event_count=event_count,
    )


def _latest_checkpoint_weights(
    weights: pl.DataFrame,
    *,
    at_or_before: date,
) -> pl.DataFrame:
    """Keep each asset's latest state across a checkpoint-date price gap."""

    return (
        weights.filter(pl.col(TIME) <= at_or_before)
        .sort([ASSET_ID, TIME])
        .group_by(ASSET_ID, maintain_order=True)
        .last()
        .select(ASSET_ID, "weight")
        .sort(ASSET_ID)
    )


def _attach_section_series(
    path: PortfolioPathChunk,
    spec: ResultSectionSpec,
    window: ResultWindow,
    metrics: dict[str, float | int | None],
    tables: dict[str, pl.DataFrame],
    *,
    annualization: int,
) -> None:
    """Attach only the diagnostic series needed by the requested section."""

    required = {
        "Summary & Coverage": ("coverage",),
        "IC & ICIR": ("ic",),
        "Quantiles": ("quantile_returns",),
        "TOP N": ("lag_analysis", "lag_returns"),
        "TOP N vs Benchmarks": (
            "benchmark_returns",
            "benchmark_coverage",
            "benchmark_performance",
            "excess_returns",
        ),
        "Spread Performance": ("spread_returns",),
    }[spec.section]
    selected = set(spec.items)
    for key in required:
        if selected and key not in selected:
            continue
        if key in {"benchmark_performance", "excess_returns"}:
            continue
        frame = path.series.get(key)
        if frame is None:
            continue
        if "time" in frame.columns:
            if key in {"benchmark_returns", "benchmark_coverage"}:
                frame = frame.join(
                    tables["returns"].select(TIME),
                    on=TIME,
                    how="inner",
                )
            else:
                frame = frame.filter(
                    pl.col("time").is_between(window.start, window.end)
                )
        tables[key] = frame
    benchmark_returns = tables.get("benchmark_returns")
    if benchmark_returns is not None and not benchmark_returns.is_empty():
        if not selected or "benchmark_performance" in selected:
            tables["benchmark_performance"] = benchmark_performance(
                benchmark_returns,
                annualization=annualization,
            )
        if not selected or "excess_returns" in selected:
            tables["excess_returns"] = top_n_excess_returns(
                tables["returns"].select(
                    TIME,
                    "gross_return",
                    "net_return",
                ),
                benchmark_returns,
            )
    ic = tables.get("ic")
    if ic is not None and "ic" in ic.columns and not ic.is_empty():
        values = ic.get_column("ic").drop_nulls()
        metrics.update(
            {
                "ic_mean": values.mean(),
                "ic_std": values.std(),
                "icir": (
                    None
                    if values.std() in {None, 0.0}
                    else float(values.mean())
                    / float(values.std())
                    * annualization**0.5
                ),
            }
        )
    coverage = tables.get("coverage")
    if (
        coverage is not None
        and "coverage_ratio" in coverage.columns
        and not coverage.is_empty()
    ):
        metrics["mean_coverage"] = coverage.get_column("coverage_ratio").mean()
        metrics["minimum_coverage"] = coverage.get_column("coverage_ratio").min()
    quantiles = tables.get("quantile_returns")
    if (
        quantiles is not None
        and "quantile" in quantiles.columns
        and not quantiles.is_empty()
    ):
        metrics["quantile_count"] = quantiles.get_column("quantile").n_unique()
    spread = tables.get("spread_returns")
    if spread is not None and "return" in spread.columns and not spread.is_empty():
        values = spread.get_column("return").drop_nulls()
        total = float((values + 1.0).product() - 1.0)
        volatility = values.std()
        metrics.update(
            {
                "spread_total_return": total,
                "spread_annualized_return": (
                    (1.0 + total) ** (annualization / len(values)) - 1.0
                ),
                "spread_sharpe": (
                    None
                    if volatility in {None, 0.0}
                    else float(values.mean())
                    / float(volatility)
                    * annualization**0.5
                ),
            }
        )


def _sparse_weight_changes(
    weights: pl.DataFrame,
    initial_weights: pl.DataFrame | None,
) -> pl.DataFrame:
    initial = (
        pl.DataFrame(schema={ASSET_ID: pl.String, "_initial": pl.Float64})
        if initial_weights is None
        else initial_weights.select(
            ASSET_ID,
            pl.col("weight").alias("_initial"),
        )
    )
    return (
        weights.sort([ASSET_ID, TIME])
        .join(initial, on=ASSET_ID, how="left")
        .with_columns(
            pl.col("weight")
            .shift(1)
            .over(ASSET_ID)
            .fill_null(pl.col("_initial"))
            .fill_null(0.0)
            .alias("_previous")
        )
        .filter(pl.col("weight") != pl.col("_previous"))
        .select(TIME, ASSET_ID, "weight")
        .sort([TIME, ASSET_ID])
    )


def _summary_metrics(summary: PerformanceSummary) -> dict[str, float]:
    return {
        "gross_total_return": summary.gross_total_return,
        "net_total_return": summary.net_total_return,
        "gross_annualized_return": summary.gross_annualized_return,
        "net_annualized_return": summary.net_annualized_return,
        "gross_sharpe": summary.gross_sharpe,
        "net_sharpe": summary.net_sharpe,
        "gross_max_drawdown": summary.gross_max_drawdown,
        "net_max_drawdown": summary.net_max_drawdown,
        "average_turnover": summary.average_turnover,
        "total_transaction_cost": summary.total_transaction_cost,
    }


__all__ = [
    "PORTFOLIO_PATH_VERSION",
    "RESULT_SECTIONS",
    "RESULT_SECTION_VERSION",
    "PortfolioPathChunk",
    "PortfolioPathIdentity",
    "PortfolioStateCheckpoint",
    "ResultSection",
    "ResultSectionSpec",
    "ResultWindow",
    "compute_result_section",
    "materialize_portfolio_path",
    "portfolio_path_from_backtest",
    "resume_portfolio_path",
]
