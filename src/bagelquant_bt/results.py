"""Result containers returned by bagelquant-bt."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from threading import Lock

import polars as pl

from .inputs import ASSET_ID, TIME


@dataclass(slots=True)
class _DeferredMarketKeys:
    """Thread-safe shared sort cache for deferred portfolio frames."""

    frame: pl.DataFrame
    _sorted: pl.DataFrame | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def sorted(self) -> pl.DataFrame:
        cached = self._sorted
        if cached is not None:
            return cached
        with self._lock:
            cached = self._sorted
            if cached is None:
                cached = self.frame.sort([ASSET_ID, TIME])
                self._sorted = cached
        return cached


@dataclass(slots=True)
class _DeferredPortfolioFrame:
    """Thread-safe, one-shot expansion of sparse portfolio state."""

    market_keys: pl.DataFrame | _DeferredMarketKeys
    state_events: pl.DataFrame
    _cached: pl.DataFrame | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def materialize(self) -> pl.DataFrame:
        cached = self._cached
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cached
            if cached is None:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=(
                            "Sortedness of columns cannot be checked when 'by' "
                            "groups provided"
                        ),
                        category=UserWarning,
                    )
                    market_keys = (
                        self.market_keys.sorted()
                        if isinstance(self.market_keys, _DeferredMarketKeys)
                        else self.market_keys.sort([ASSET_ID, TIME])
                    )
                    cached = (
                        market_keys
                        .join_asof(
                            self.state_events.sort([ASSET_ID, TIME]),
                            on=TIME,
                            by=ASSET_ID,
                            strategy="backward",
                        )
                        .drop_nulls("weight")
                        .select(TIME, ASSET_ID, "weight")
                        .sort([TIME, ASSET_ID])
                    )
                self._cached = cached
        return cached


@dataclass(frozen=True, slots=True)
class TransactionCostBreakdown:
    """Daily transaction cost details."""

    data: pl.DataFrame


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """High-level performance metrics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    gross_total_return: float
    net_total_return: float
    gross_annualized_return: float
    net_annualized_return: float
    gross_annualized_volatility: float
    net_annualized_volatility: float
    gross_sharpe: float
    net_sharpe: float
    gross_max_drawdown: float
    net_max_drawdown: float
    hit_rate: float
    average_turnover: float
    total_transaction_cost: float
    final_gross_value: float
    final_net_value: float


@dataclass(frozen=True, slots=True, repr=False)
class BacktestResult:
    """Portfolio backtest result with gross and net return paths."""

    weights: pl.DataFrame
    asset_returns: pl.DataFrame
    returns: pl.DataFrame
    value: pl.DataFrame
    turnover: pl.DataFrame
    transaction_costs: TransactionCostBreakdown
    summary: PerformanceSummary
    performance: pl.DataFrame
    annualization: int
    coverage: pl.DataFrame
    missing_price_keys: pl.DataFrame
    price_gaps: pl.DataFrame
    unexecuted_weight_keys: pl.DataFrame
    execution_blocks: pl.DataFrame
    target_weights: pl.DataFrame
    execution_event_count: int = 0

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in {"weights", "target_weights"} and isinstance(
            value, _DeferredPortfolioFrame
        ):
            frame = value.materialize()
            object.__setattr__(self, name, frame)
            return frame
        return value

    def __repr__(self) -> str:
        weights = object.__getattribute__(self, "weights")
        targets = object.__getattribute__(self, "target_weights")
        weights_state = (
            "<deferred>"
            if isinstance(weights, _DeferredPortfolioFrame)
            else f"DataFrame{weights.shape}"
        )
        targets_state = (
            "<deferred>"
            if isinstance(targets, _DeferredPortfolioFrame)
            else f"DataFrame{targets.shape}"
        )
        return (
            "BacktestResult("
            f"weights={weights_state}, target_weights={targets_state}, "
            f"returns=DataFrame{self.returns.shape}, "
            f"value=DataFrame{self.value.shape})"
        )


@dataclass(frozen=True, slots=True)
class FactorEvaluationResult:
    """Factor diagnostics and derived TOP N backtest."""

    factor: pl.DataFrame
    forward_returns: pl.DataFrame
    ic: pl.DataFrame
    ic_summary: pl.DataFrame
    ic_mean: float
    ic_std: float
    icir: float
    ic_annualization: int
    quantile_returns: pl.DataFrame
    spread_returns: pl.DataFrame
    top_n_weights: pl.DataFrame
    top_n_backtest: BacktestResult
    spread_weights: pl.DataFrame
    spread_backtest: BacktestResult | None
    lag_analysis: pl.DataFrame
    lag_returns: pl.DataFrame
    ic_decay: pl.DataFrame
    coverage: pl.DataFrame
    missing_price_keys: pl.DataFrame
    benchmark_returns: pl.DataFrame
    benchmark_coverage: pl.DataFrame
    benchmark_performance: pl.DataFrame
    excess_returns: pl.DataFrame


SignalEvaluationResult = FactorEvaluationResult
