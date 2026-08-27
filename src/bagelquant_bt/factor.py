"""Factor evaluation routines."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl
from bagelquant_core import Domain, PredictionPanel

from ._quantiles import sort_quantile_frame
from .benchmarks import (
    DEFAULT_BENCHMARK,
    benchmark_performance,
    build_universe_benchmark_returns,
    top_n_excess_returns,
    validate_benchmark_returns,
)
from .comparison import run_total_return_weight_paths
from .config import BacktestConfig
from .engine import (
    _backtest_weight_frames_with_forward_returns,
    _CompactBacktestResult,
    _prepare_sparse_market_context,
    _require_config,
    _run_sparse_compact_backtests,
    _SparseMarketContext,
)
from .exceptions import InputValidationError
from .inputs import (
    ASSET_ID,
    TIME,
    asset_coverage,
    missing_price_keys,
    validate_execution_availability,
    validate_factor,
    validate_panel_frame,
    validate_prices,
)
from .policy import ScheduledPrediction
from .portfolio import EqualWeightPolicy
from .results import BacktestResult, FactorEvaluationResult
from .returns import PreparedPriceData, _prepare_price_data, prepare_price_data

FACTOR_LAGS = (0, 1, 2, 3, 4, 5, 10, 20, 30, 60)


@dataclass(frozen=True, slots=True)
class PreparedFactorMarketData:
    """Validated market inputs reusable across factor evaluations."""

    prices: pl.DataFrame
    forward_returns: pl.DataFrame
    price_data: PreparedPriceData | None = None
    total_return_prices: pl.DataFrame | None = None


def prepare_factor_market_data(prices: pl.DataFrame) -> PreparedFactorMarketData:
    """Validate prices and calculate forward returns once for a factor batch."""

    aligned_prices = validate_prices(prices)
    price_data = _prepare_price_data(aligned_prices, inputs_sorted=True)
    return PreparedFactorMarketData(
        prices=aligned_prices,
        forward_returns=price_data.forward_returns,
        price_data=price_data,
    )


def run_factor_evaluation(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig | None = None,
    market_data: PreparedFactorMarketData | None = None,
    coverage_universe: pl.DataFrame | None = None,
    benchmark_universe: pl.DataFrame | None = None,
    execution_availability: pl.DataFrame | None = None,
    benchmark_returns: pl.DataFrame | None = None,
    benchmark_coverage: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
) -> FactorEvaluationResult:
    """Evaluate a factor score frame against forward returns and membership."""

    resolved_config = _require_config(config)
    aligned_factor = validate_factor(factor)
    prepared = market_data or prepare_factor_market_data(prices)
    return evaluate_factor_frame(
        aligned_factor,
        prepared.prices,
        config=resolved_config,
        market_data=prepared,
        coverage_universe=coverage_universe,
        benchmark_universe=benchmark_universe,
        execution_availability=execution_availability,
        benchmark_returns=benchmark_returns,
        benchmark_coverage=benchmark_coverage,
        slippage_rates=slippage_rates,
    )


def run_prediction_evaluation(
    signals: ScheduledPrediction,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig | None = None,
    market_data: PreparedFactorMarketData | None = None,
    evaluation_returns: pl.DataFrame | None = None,
    portfolio_policy: object | None = None,
    portfolio_inputs: Mapping[str, object] | None = None,
    coverage_universe: pl.DataFrame | None = None,
    benchmark_universe: pl.DataFrame | None = None,
    execution_availability: pl.DataFrame | None = None,
    benchmark_returns: pl.DataFrame | None = None,
    benchmark_coverage: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
    total_return_prices: pl.DataFrame | None = None,
    include_synthetic_diagnostics: bool = True,
) -> FactorEvaluationResult:
    """Evaluate executable signals against returns through the next signal."""

    resolved_config = _require_config(config)
    if not isinstance(signals, ScheduledPrediction):
        raise TypeError("run_prediction_evaluation requires a ScheduledPrediction")
    signal_frame = signals.prediction.collect(dense=False).rename({"value": "signal"})
    aligned = validate_panel_frame(
        signal_frame, label="signals", value_columns=("signal",)
    )
    factor = aligned.select(TIME, ASSET_ID, pl.col("signal").alias("factor"))
    prepared = market_data or prepare_factor_market_data(prices)
    resolved_evaluation_returns = (
        _validate_prepared_evaluation_returns(
            evaluation_returns,
            factor=factor,
            prices=prepared.prices,
        )
        if evaluation_returns is not None
        else prediction_forward_returns(factor, prepared.prices)
    )
    return evaluate_factor_frame(
        factor,
        prepared.prices,
        config=resolved_config,
        market_data=prepared,
        coverage_universe=coverage_universe,
        benchmark_universe=benchmark_universe,
        evaluation_returns=resolved_evaluation_returns,
        lag_return_provider=lambda lagged: prediction_forward_returns(
            lagged, prepared.prices
        ),
        portfolio_policy=portfolio_policy,
        portfolio_inputs=portfolio_inputs,
        execution_availability=execution_availability,
        benchmark_returns=benchmark_returns,
        benchmark_coverage=benchmark_coverage,
        slippage_rates=slippage_rates,
        total_return_prices=total_return_prices,
        include_synthetic_diagnostics=include_synthetic_diagnostics,
    )


def materialize_prediction_diagnostics(
    signals: ScheduledPrediction,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig | None = None,
    market_data: PreparedFactorMarketData | None = None,
    execution_availability: pl.DataFrame | None = None,
    include_quantiles: bool = False,
    include_spread: bool = False,
    include_lags: bool = False,
    slippage_rates: pl.DataFrame | None = None,
) -> Mapping[str, pl.DataFrame]:
    """Build only requested heavyweight signal diagnostic paths."""

    if not (include_quantiles or include_spread or include_lags):
        return {}
    resolved_config = _require_config(config)
    if not isinstance(signals, ScheduledPrediction):
        raise TypeError(
            "materialize_prediction_diagnostics requires a ScheduledPrediction"
        )
    signal_frame = signals.prediction.collect(dense=False).rename({"value": "signal"})
    aligned = validate_panel_frame(
        signal_frame,
        label="signals",
        value_columns=("signal",),
    )
    factor = aligned.select(TIME, ASSET_ID, pl.col("signal").alias("factor"))
    return materialize_factor_diagnostics(
        factor,
        prices,
        config=resolved_config,
        market_data=market_data,
        execution_availability=execution_availability,
        include_quantiles=include_quantiles,
        include_spread=include_spread,
        include_lags=include_lags,
        slippage_rates=slippage_rates,
    )


def materialize_factor_diagnostics(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig | None = None,
    market_data: PreparedFactorMarketData | None = None,
    execution_availability: pl.DataFrame | None = None,
    include_quantiles: bool = False,
    include_spread: bool = False,
    include_lags: bool = False,
    slippage_rates: pl.DataFrame | None = None,
) -> Mapping[str, pl.DataFrame]:
    """Build requested heavyweight diagnostics from a cached factor frame."""

    if not (include_quantiles or include_spread or include_lags):
        return {}
    resolved_config = _require_config(config)
    factor = validate_factor(factor)
    prepared = market_data or prepare_factor_market_data(prices)
    resolved_execution_availability = (
        None
        if execution_availability is None
        else validate_execution_availability(execution_availability)
    )
    market_context = _prepare_sparse_market_context(
        prepared.prices,
        prepared.forward_returns,
        resolved_execution_availability,
    )
    result: dict[str, pl.DataFrame] = {}
    diagnostic_quantile_weights: dict[str, pl.DataFrame] = {}
    if include_quantiles or include_spread:
        all_quantile_weights = quantile_equal_weights(
            factor,
            quantiles=resolved_config.quantiles,
        )
        requested_labels = (
            tuple(all_quantile_weights)
            if include_quantiles
            else ("q1", f"q{resolved_config.quantiles}")
        )
        diagnostic_quantile_weights = {
            label: all_quantile_weights[label] for label in requested_labels
        }
    if include_lags:
        lag_analysis, lag_returns, _ = _lag_outputs(
            factor,
            prepared.prices,
            config=resolved_config,
            lags=FACTOR_LAGS,
            forward_returns=prepared.forward_returns,
            price_gaps=(
                None if prepared.price_data is None else prepared.price_data.price_gaps
            ),
            execution_availability=resolved_execution_availability,
            execution_availability_validated=True,
            market_context=market_context,
            additional_weight_frames=diagnostic_quantile_weights,
            slippage_rates=slippage_rates,
        )
        result["lag_analysis"] = lag_analysis
        result["lag_returns"] = lag_returns
    if diagnostic_quantile_weights:
        quantile_returns = (
            _batched_quantile_gross_returns(
                diagnostic_quantile_weights,
                prepared.prices,
                prepared.forward_returns,
                execution_availability=resolved_execution_availability,
                retry_blocked=resolved_config.retry_blocked_orders,
            )
            .rename({"gross_return": "return"})
            .with_columns(
                (
                    (1.0 + pl.col("return").fill_null(0.0)).cum_prod().over("quantile")
                    - 1.0
                ).alias("cumulative_return")
            )
        )
        if include_quantiles:
            result["quantile_returns"] = quantile_returns
        if include_spread:
            result["spread_returns"] = _spread_returns(
                quantile_returns,
                resolved_config.quantiles,
            ).rename({"spread_return": "return"})
    return result


def _validate_prepared_evaluation_returns(
    frame: pl.DataFrame,
    *,
    factor: pl.DataFrame,
    prices: pl.DataFrame,
) -> pl.DataFrame:
    """Validate that cached returns belong to this signal schedule and market."""

    normalized = validate_panel_frame(
        frame,
        label="evaluation_returns",
        value_columns=("forward_return",),
    ).select(TIME, ASSET_ID, "forward_return")
    unexpected_times = (
        normalized.select(TIME)
        .unique()
        .join(
            factor.select(TIME).unique(),
            on=TIME,
            how="anti",
        )
    )
    if unexpected_times.height:
        raise InputValidationError(
            "evaluation_returns contains dates outside the current signal schedule"
        )
    unexpected_signal_keys = normalized.select(TIME, ASSET_ID).join(
        factor.select(TIME, ASSET_ID),
        on=[TIME, ASSET_ID],
        how="anti",
    )
    if unexpected_signal_keys.height:
        raise InputValidationError(
            "evaluation_returns contains asset keys outside the current signals"
        )
    unexpected_keys = normalized.select(TIME, ASSET_ID).join(
        prices.select(TIME, ASSET_ID),
        on=[TIME, ASSET_ID],
        how="anti",
    )
    if unexpected_keys.height:
        raise InputValidationError(
            "evaluation_returns contains asset keys absent from prepared prices"
        )
    return normalized


def evaluate_factor_frame(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    market_data: PreparedFactorMarketData | None = None,
    coverage_universe: pl.DataFrame | None = None,
    benchmark_universe: pl.DataFrame | None = None,
    evaluation_returns: pl.DataFrame | None = None,
    lag_return_provider: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    portfolio_policy: object | None = None,
    portfolio_inputs: Mapping[str, object] | None = None,
    execution_availability: pl.DataFrame | None = None,
    benchmark_returns: pl.DataFrame | None = None,
    benchmark_coverage: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
    total_return_prices: pl.DataFrame | None = None,
    include_synthetic_diagnostics: bool = True,
) -> FactorEvaluationResult:
    """Evaluate an already materialized factor score frame."""

    aligned_factor = validate_factor(factor)
    prepared = market_data or prepare_factor_market_data(prices)
    resolved_execution_availability = (
        None
        if execution_availability is None
        else validate_execution_availability(execution_availability)
    )
    aligned_prices = prepared.prices
    price_data = prepared.price_data or _prepare_price_data(
        aligned_prices, inputs_sorted=True
    )
    coverage = asset_coverage(
        aligned_factor,
        aligned_prices,
        asset_count_column="factor_signal_asset_count",
        coverage_universe=coverage_universe,
    )
    missing_keys = missing_price_keys(aligned_factor, aligned_prices)
    factor = (
        aligned_factor.join(
            aligned_prices.select(TIME, ASSET_ID), on=[TIME, ASSET_ID], how="inner"
        )
        .select(TIME, ASSET_ID, "factor")
        .sort([TIME, ASSET_ID])
    )
    forward_returns = prepared.forward_returns
    metric_returns = (
        evaluation_returns if evaluation_returns is not None else forward_returns
    )
    if factor.is_empty():
        raise InputValidationError("at least two overlapping price times are required")
    market_context = _prepare_sparse_market_context(
        aligned_prices,
        forward_returns,
        resolved_execution_availability,
    )

    ic = information_coefficients(factor, metric_returns)
    ic_summary = summarize_ic(ic, annualization=config.resolved_ic_annualization)
    values = np.array(ic["spearman_ic"].drop_nulls(), dtype=float)
    ic_std = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
    ic_mean = float(np.mean(values)) if len(values) else math.nan
    icir = (
        ic_mean / ic_std * math.sqrt(config.resolved_ic_annualization)
        if ic_std != 0 and not math.isnan(ic_std)
        else math.nan
    )
    top_n_weights = _policy_weights(
        factor,
        config,
        portfolio_policy,
        aligned_prices,
        portfolio_inputs,
    )
    if top_n_weights.is_empty():
        raise InputValidationError("portfolio policy produced no valid target weights")
    spread_weights = pl.DataFrame(
        schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64}
    )
    quantile_returns = pl.DataFrame(
        schema={
            TIME: pl.Date,
            "quantile": pl.String,
            "return": pl.Float64,
            "cumulative_return": pl.Float64,
        }
    )
    spread_returns = pl.DataFrame(schema={TIME: pl.Date, "spread_return": pl.Float64})
    lag_analysis = pl.DataFrame()
    lag_returns = pl.DataFrame()
    ic_decay = pl.DataFrame()
    primary_compact_results: Mapping[str, _CompactBacktestResult] | None = None
    if include_synthetic_diagnostics:
        quantile_weights = quantile_equal_weights(
            factor,
            quantiles=config.quantiles,
        )
        spread_weights = spread_quantile_weights(
            factor,
            quantiles=config.quantiles,
        )
        lag_analysis, lag_returns, primary_compact_results = _lag_outputs(
            factor,
            aligned_prices,
            config=config,
            lags=FACTOR_LAGS,
            forward_returns=forward_returns,
            price_gaps=price_data.price_gaps,
            execution_availability=resolved_execution_availability,
            execution_availability_validated=True,
            market_context=market_context,
            additional_weight_frames={
                "top_n": top_n_weights,
                **({"spread": spread_weights} if spread_weights.height else {}),
                **{
                    f"quantile:{label}": weights
                    for label, weights in quantile_weights.items()
                },
            },
            slippage_rates=slippage_rates,
        )
        quantile_returns = (
            _batched_quantile_gross_returns(
                quantile_weights,
                aligned_prices if total_return_prices is None else total_return_prices,
                forward_returns,
                execution_availability=resolved_execution_availability,
                retry_blocked=config.retry_blocked_orders,
            )
            .rename({"gross_return": "return"})
            .with_columns(
                (
                    (1.0 + pl.col("return").fill_null(0.0)).cum_prod().over("quantile")
                    - 1.0
                ).alias("cumulative_return")
            )
        )
        spread_returns = _spread_returns(quantile_returns, config.quantiles)
        ic_decay = factor_ic_decay(
            factor,
            metric_returns,
            trading_sessions=_trading_sessions(aligned_prices),
            return_provider=lag_return_provider,
            lags=FACTOR_LAGS,
        )
    primary_backtests = _backtest_weight_frames_with_forward_returns(
        {
            "top_n": top_n_weights,
            **({"spread": spread_weights} if spread_weights.height else {}),
        },
        aligned_prices,
        forward_returns,
        config=config,
        price_gaps=price_data.price_gaps,
        execution_availability=resolved_execution_availability,
        execution_availability_validated=True,
        market_context=market_context,
        precomputed_compact_results=primary_compact_results,
        slippage_rates=slippage_rates,
    )
    top_n_backtest = primary_backtests["top_n"]
    spread_backtest = primary_backtests.get("spread")
    default_benchmark, default_coverage = build_universe_benchmark_returns(
        forward_returns,
        universe=(
            benchmark_universe if benchmark_universe is not None else coverage_universe
        ),
        name=DEFAULT_BENCHMARK,
    )
    resolved_benchmarks = default_benchmark
    resolved_coverage = default_coverage
    if benchmark_returns is not None:
        external = validate_benchmark_returns(benchmark_returns)
        if DEFAULT_BENCHMARK in set(external.get_column("benchmark")):
            raise InputValidationError(
                f"external benchmark name conflicts with {DEFAULT_BENCHMARK!r}"
            )
        resolved_benchmarks = pl.concat([default_benchmark, external]).sort(
            ["benchmark", TIME]
        )
        external_coverage = (
            _validate_benchmark_coverage(benchmark_coverage)
            if benchmark_coverage is not None
            else _benchmark_return_coverage(external)
        )
        resolved_coverage = pl.concat([default_coverage, external_coverage]).sort(
            ["benchmark", TIME]
        )
    benchmark_metrics = benchmark_performance(
        resolved_benchmarks, annualization=config.annualization
    )
    excess_returns = top_n_excess_returns(top_n_backtest.returns, resolved_benchmarks)

    return FactorEvaluationResult(
        factor=factor,
        forward_returns=metric_returns,
        ic=ic,
        ic_summary=ic_summary,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        ic_annualization=config.resolved_ic_annualization,
        quantile_returns=quantile_returns,
        spread_returns=spread_returns,
        top_n_weights=top_n_weights,
        top_n_backtest=top_n_backtest,
        spread_weights=spread_weights,
        spread_backtest=spread_backtest,
        lag_analysis=lag_analysis,
        lag_returns=lag_returns,
        ic_decay=ic_decay,
        coverage=coverage,
        missing_price_keys=missing_keys,
        benchmark_returns=resolved_benchmarks,
        benchmark_coverage=resolved_coverage,
        benchmark_performance=benchmark_metrics,
        excess_returns=excess_returns,
    )


def _validate_benchmark_coverage(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        TIME,
        "benchmark",
        "expected_count",
        "observed_count",
        "coverage_ratio",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError(
            f"benchmark_coverage is missing required columns: {missing}"
        )
    normalized = (
        frame.select(*sorted(required))
        .with_columns(
            pl.col(TIME).cast(pl.Date, strict=False),
            pl.col("benchmark").cast(pl.String),
            pl.col("expected_count").cast(pl.Int64),
            pl.col("observed_count").cast(pl.Int64),
            pl.col("coverage_ratio").cast(pl.Float64),
        )
        .drop_nulls()
    )
    if normalized.select(pl.struct(TIME, "benchmark").is_duplicated().any()).item():
        raise InputValidationError(
            "benchmark_coverage must be unique by (time, benchmark)"
        )
    return normalized.select(
        TIME,
        "benchmark",
        "expected_count",
        "observed_count",
        "coverage_ratio",
    )


def _benchmark_return_coverage(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.select(TIME, "benchmark")
        .with_columns(
            pl.lit(1, dtype=pl.Int64).alias("expected_count"),
            pl.lit(1, dtype=pl.Int64).alias("observed_count"),
            pl.lit(1.0).alias("coverage_ratio"),
        )
        .select(
            TIME,
            "benchmark",
            "expected_count",
            "observed_count",
            "coverage_ratio",
        )
    )


def prediction_forward_returns(
    signals: pl.DataFrame,
    prices: pl.DataFrame,
) -> pl.DataFrame:
    """Return each signal's asset return through its next executable signal."""

    schedule = (
        signals.select(TIME)
        .unique()
        .sort(TIME)
        .with_columns(pl.col(TIME).shift(-1).alias("next_time"))
        .drop_nulls("next_time")
    )
    if schedule.is_empty():
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, "forward_return": pl.Float64}
        )
    starts = signals.join(schedule, on=TIME, how="inner")
    end_prices = prices.select(
        pl.col(TIME).alias("next_time"),
        ASSET_ID,
        pl.col("price").alias("next_price"),
    )
    return (
        starts.join(
            prices.select(TIME, ASSET_ID, pl.col("price").alias("start_price")),
            on=[TIME, ASSET_ID],
            how="inner",
        )
        .join(end_prices, on=["next_time", ASSET_ID], how="inner")
        .select(
            TIME,
            ASSET_ID,
            (pl.col("next_price") / pl.col("start_price") - 1.0).alias(
                "forward_return"
            ),
        )
        .sort([TIME, ASSET_ID])
    )


def _policy_weights(
    factor: pl.DataFrame,
    config: BacktestConfig,
    policy: object | None,
    prices: pl.DataFrame,
    inputs: Mapping[str, object] | None,
) -> pl.DataFrame:
    selected = policy or EqualWeightPolicy(config.top_n)
    build = getattr(selected, "build", None)
    if build is None:
        raise TypeError("weight_policy must define build(PredictionPanel, ...)")
    prediction_frame = factor.select(
        TIME,
        ASSET_ID,
        pl.col("factor").alias("value"),
    )
    domain = Domain(
        calendar=prediction_frame.get_column(TIME).unique().sort(),
        universe=prediction_frame.get_column(ASSET_ID).unique().sort(),
    )
    output = build(
        PredictionPanel.from_domain(prediction_frame, domain, name="prediction"),
        prices=prices,
        config=config,
        **dict(inputs or {}),
    )
    return output.weights.collect(dense=False).rename({"value": "weight"})


def information_coefficients(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Compute daily Pearson and Spearman cross-sectional IC."""

    paired = factor.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
    if paired.is_empty():
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "pearson_ic": pl.Float64,
                "spearman_ic": pl.Float64,
            }
        )
    times = paired.select(TIME).unique()
    values = paired.drop_nulls(["factor", "forward_return"])
    if values.is_empty():
        ic = pl.DataFrame(
            schema={
                TIME: paired.schema[TIME],
                "pearson_ic": pl.Float64,
                "spearman_ic": pl.Float64,
            }
        )
    else:
        ic = (
            values.with_columns(
                pl.col("factor").rank("average").over(TIME).alias("_factor_rank"),
                pl.col("forward_return")
                .rank("average")
                .over(TIME)
                .alias("_return_rank"),
            )
            .group_by(TIME)
            .agg(
                _corr_expr("factor", "forward_return").alias("pearson_ic"),
                _corr_expr("_factor_rank", "_return_rank").alias("spearman_ic"),
            )
        )
    return times.join(ic, on=TIME, how="left").sort(TIME)


def information_coefficient(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    method: str = "spearman",
) -> pl.DataFrame:
    """Compute daily cross-sectional IC."""

    return _information_coefficient_values(
        factor,
        forward_returns,
        method=method,
        output_column="ic",
    )


def summarize_ic(ic: pl.DataFrame, *, annualization: int) -> pl.DataFrame:
    """Summarize annualized Pearson and Spearman IC information ratios."""

    rows: list[dict[str, object]] = []
    for method, column in (("pearson", "pearson_ic"), ("spearman", "spearman_ic")):
        values = np.array(ic[column].drop_nulls(), dtype=float)
        std = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        mean = float(np.mean(values)) if len(values) else math.nan
        icir = (
            mean / std * math.sqrt(annualization)
            if std != 0 and not math.isnan(std)
            else math.nan
        )
        rows.append({"method": method, "mean": mean, "std": std, "icir": icir})
    return pl.DataFrame(rows)


def _information_coefficient_values(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    method: str,
    output_column: str,
) -> pl.DataFrame:
    """Compute daily cross-sectional IC for one method."""

    paired = factor.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
    if paired.is_empty():
        return pl.DataFrame(schema={TIME: pl.Date, output_column: pl.Float64})
    times = paired.select(TIME).unique()
    if method == "pearson":
        left = "factor"
        right = "forward_return"
        values = paired.drop_nulls([left, right])
    elif method == "spearman":
        left = "_factor_rank"
        right = "_return_rank"
        values = paired.drop_nulls(["factor", "forward_return"]).with_columns(
            pl.col("factor").rank("average").over(TIME).alias(left),
            pl.col("forward_return").rank("average").over(TIME).alias(right),
        )
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")
    if values.is_empty():
        ic = pl.DataFrame(schema={TIME: paired.schema[TIME], output_column: pl.Float64})
    else:
        ic = values.group_by(TIME).agg(_corr_expr(left, right).alias(output_column))
    return times.join(ic, on=TIME, how="left").sort(TIME)


def factor_quantile_returns(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    """Compute equal-weight daily returns by factor quantile."""

    paired = factor.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
    if paired.is_empty():
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "quantile": pl.String,
                "return": pl.Float64,
                "cumulative_return": pl.Float64,
                "is_bankrupt": pl.Boolean,
                "bankruptcy_event": pl.Boolean,
            }
        )
    quantile_grid = _quantile_grid(paired.select(TIME).unique(), quantiles)
    bucket_returns = (
        _quantile_bucket_frame(
            paired.drop_nulls(["factor", "forward_return"]),
            quantiles=quantiles,
        )
        .group_by(TIME, "bucket")
        .agg(pl.col("forward_return").mean().alias("return"))
        .with_columns(
            (pl.lit("q") + pl.col("bucket").cast(pl.String)).alias("quantile")
        )
        .select(TIME, "quantile", "return")
    )
    returns = quantile_grid.join(
        bucket_returns,
        on=[TIME, "quantile"],
        how="left",
    )
    returns = sort_quantile_frame(returns, before=(TIME,))
    return returns.with_columns(
        (
            (1.0 + pl.col("return").fill_null(0.0)).cum_prod().over("quantile") - 1.0
        ).alias("cumulative_return")
    )


def traded_factor_quantile_returns(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    quantiles: int,
    execution_availability: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute quantile return series by trading quantile portfolios."""

    return _traded_factor_quantile_returns_with_forward_returns(
        factor,
        prices,
        config=config,
        quantiles=quantiles,
        forward_returns=None,
        execution_availability=execution_availability,
    )


def _traded_factor_quantile_returns_with_forward_returns(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    quantiles: int,
    forward_returns: pl.DataFrame | None,
    price_gaps: pl.DataFrame | None = None,
    execution_availability: pl.DataFrame | None = None,
    execution_availability_validated: bool = False,
) -> pl.DataFrame:
    """Compute daily held-portfolio quantile returns from signal snapshots.

    Quantile memberships and weights change only on factor snapshot dates.
    The backtest expands each snapshot across daily price returns until the next
    snapshot, so a monthly signal remains a monthly-rebalanced portfolio.
    """

    quantile_weights = quantile_equal_weights(
        factor,
        quantiles=quantiles,
    )
    nonempty_weights = [
        weights for weights in quantile_weights.values() if not weights.is_empty()
    ]
    resolved_forward_returns = (
        prepare_price_data(prices).forward_returns
        if forward_returns is None
        else forward_returns
    )
    resolved_availability = (
        validate_execution_availability(execution_availability)
        if execution_availability is not None and not execution_availability_validated
        else execution_availability
    )
    if not nonempty_weights:
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "quantile": pl.String,
                "return": pl.Float64,
                "cumulative_return": pl.Float64,
            }
        )
    returns = _batched_quantile_gross_returns(
        quantile_weights,
        prices,
        resolved_forward_returns,
        execution_availability=resolved_availability,
        retry_blocked=config.retry_blocked_orders,
    ).rename({"gross_return": "return"})
    return returns.with_columns(
        (
            (1.0 + pl.col("return").fill_null(0.0)).cum_prod().over("quantile") - 1.0
        ).alias("cumulative_return")
    )


def _quantile_returns_from_compact_results(
    quantile_weights: Mapping[str, pl.DataFrame],
    compact_results: Mapping[str, _CompactBacktestResult],
) -> pl.DataFrame:
    """Restore the public quantile table from unified batch results."""

    if not compact_results:
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "quantile": pl.String,
                "return": pl.Float64,
                "cumulative_return": pl.Float64,
            }
        )
    return_frames = [
        compact_results[label].returns.select(
            TIME,
            pl.lit(label).alias("quantile"),
            pl.col("gross_return").alias("return"),
            "is_bankrupt",
            "bankruptcy_event",
        )
        for label in quantile_weights
        if label in compact_results
    ]
    times = return_frames[0].select(TIME)
    grid = times.join(
        pl.DataFrame(
            {"quantile": list(quantile_weights)},
            schema={"quantile": pl.String},
        ),
        how="cross",
    )
    returns = grid.join(
        pl.concat(return_frames),
        on=[TIME, "quantile"],
        how="left",
    ).with_columns(
        pl.col("return").fill_null(0.0),
        pl.col("is_bankrupt").fill_null(False),
        pl.col("bankruptcy_event").fill_null(False),
    )
    return sort_quantile_frame(returns, before=(TIME,)).with_columns(
        ((1.0 + pl.col("return")).cum_prod().over("quantile") - 1.0).alias(
            "cumulative_return"
        )
    )


def _batched_quantile_gross_returns(
    quantile_weights: Mapping[str, pl.DataFrame],
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    execution_availability: pl.DataFrame | None,
    retry_blocked: bool,
) -> pl.DataFrame:
    """Evaluate quantiles as execution-date unit portfolios held to rebalance."""

    del forward_returns  # Returns are implied by total-return index ratios.
    paths = run_total_return_weight_paths(
        quantile_weights,
        (
            prices.rename({"price": "total_return_price"})
            if "price" in prices.columns
            else prices
        ),
        execution_availability=execution_availability,
        retry_blocked=retry_blocked,
    )
    if paths.is_empty():
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "quantile": pl.String,
                "gross_return": pl.Float64,
            }
        )
    # Public quantile returns retain their historical forward-return label:
    # the row at t contains the realized total return from t to t+1.  The
    # performance ledger itself records that return when t+1 is observed.
    returns = (
        paths.sort("portfolio", TIME)
        .with_columns(pl.col("gross_return").shift(-1).over("portfolio"))
        .drop_nulls("gross_return")
        .select(
            TIME,
            pl.col("portfolio").alias("quantile"),
            "gross_return",
        )
    )
    return sort_quantile_frame(returns, before=(TIME,))


def _quantile_execution_corrections(
    assignments: pl.DataFrame,
    market_returns: pl.DataFrame,
    execution_availability: pl.DataFrame | None,
    *,
    retry_blocked: bool,
) -> pl.DataFrame:
    """Return sparse corrections between desired and constrained holdings."""

    empty = pl.DataFrame(
        schema={
            TIME: pl.Date,
            "quantile": pl.String,
            "_correction": pl.Float64,
        }
    )
    if execution_availability is None or execution_availability.is_empty():
        return empty

    transitions = assignments.with_columns(
        pl.col("quantile").shift(1).over(ASSET_ID).alias("_previous_quantile")
    )
    target_candidates = pl.concat(
        [
            transitions.drop_nulls("quantile").select(
                TIME, ASSET_ID, "quantile", "weight"
            ),
            transitions.filter(
                pl.col("_previous_quantile").is_not_null()
                & (
                    pl.col("quantile").is_null()
                    | (pl.col("quantile") != pl.col("_previous_quantile"))
                )
            ).select(
                TIME,
                ASSET_ID,
                pl.col("_previous_quantile").alias("quantile"),
                pl.lit(0.0).alias("weight"),
            ),
        ]
    ).sort(["quantile", ASSET_ID, TIME])
    target_events = (
        target_candidates.with_columns(
            pl.col("weight")
            .shift(1)
            .over("quantile", ASSET_ID)
            .fill_null(0.0)
            .alias("_previous_weight")
        )
        .filter(pl.col("weight") != pl.col("_previous_weight"))
        .drop("_previous_weight")
    )
    active_pairs = target_events.select("quantile", ASSET_ID).unique()
    rule_keys = execution_availability.select(TIME, ASSET_ID).join(
        active_pairs, on=ASSET_ID, how="inner"
    )
    market_keys = market_returns.select(TIME, ASSET_ID)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        after_rules = (
            execution_availability.select(pl.col(TIME).alias("_rule_time"), ASSET_ID)
            .sort([ASSET_ID, "_rule_time"])
            .join_asof(
                market_keys.select(pl.col(TIME).alias("_next_time"), ASSET_ID).sort(
                    [ASSET_ID, "_next_time"]
                ),
                left_on="_rule_time",
                right_on="_next_time",
                by=ASSET_ID,
                strategy="forward",
                allow_exact_matches=False,
            )
            .drop_nulls("_next_time")
            .select(pl.col("_next_time").alias(TIME), ASSET_ID)
            .join(active_pairs, on=ASSET_ID, how="inner")
        )
    event_keys = (
        pl.concat(
            [
                target_events.select(TIME, ASSET_ID, "quantile").with_columns(
                    pl.lit(True).alias("_target_changed")
                ),
                rule_keys.with_columns(pl.lit(False).alias("_target_changed")),
                after_rules.with_columns(pl.lit(False).alias("_target_changed")),
            ]
        )
        .group_by(TIME, ASSET_ID, "quantile")
        .agg(pl.col("_target_changed").any())
        .sort(["quantile", ASSET_ID, TIME])
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        events = event_keys.join_asof(
            target_events,
            on=TIME,
            by=["quantile", ASSET_ID],
            strategy="backward",
        )
    events = (
        events.drop_nulls("weight")
        .join(
            execution_availability.rename(
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
        .sort([TIME, "quantile", ASSET_ID])
    )
    if events.is_empty():
        return empty

    state_lookup = (
        events.select("quantile", ASSET_ID)
        .unique()
        .sort(["quantile", ASSET_ID])
        .with_row_index("_state_index")
    )
    events = events.join(state_lookup, on=["quantile", ASSET_ID], how="left")
    executed = np.zeros(state_lookup.height, dtype=np.float64)
    state_indices = events.get_column("_state_index").to_numpy()
    targets = events.get_column("weight").to_numpy()
    has_rule = events.get_column("_has_rule").to_numpy()
    target_changed = events.get_column("_target_changed").to_numpy()
    can_buy = events.get_column("_can_buy").fill_null(True).to_numpy()
    can_sell = events.get_column("_can_sell").fill_null(True).to_numpy()
    resolved = np.empty(events.height, dtype=np.float64)
    cancelled_targets = np.zeros(state_lookup.height, dtype=np.bool_)
    for position in range(events.height):
        state_index = int(state_indices[position])
        retained = float(executed[state_index])
        target = float(targets[position])
        if bool(target_changed[position]):
            cancelled_targets[state_index] = False
        delta = target - retained
        blocked = has_rule[position] and (
            (delta > 0.0 and not can_buy[position])
            or (delta < 0.0 and not can_sell[position])
        )
        if blocked:
            resolved[position] = retained
            if not retry_blocked:
                cancelled_targets[state_index] = True
        elif (
            not retry_blocked
            and cancelled_targets[state_index]
            and not target_changed[position]
        ):
            resolved[position] = retained
        else:
            resolved[position] = target
            executed[state_index] = target

    difference_events = (
        events.select(TIME, ASSET_ID, "quantile", "weight")
        .with_columns(
            (pl.Series("_executed", resolved) - pl.col("weight")).alias(
                "_weight_difference"
            )
        )
        .select(TIME, ASSET_ID, "quantile", "_weight_difference")
        .sort(["quantile", ASSET_ID, TIME])
    )
    affected_pairs = (
        difference_events.filter(pl.col("_weight_difference") != 0.0)
        .select("quantile", ASSET_ID)
        .unique()
    )
    if affected_pairs.is_empty():
        return empty
    correction_market = market_returns.join(
        affected_pairs, on=ASSET_ID, how="inner"
    ).sort(["quantile", ASSET_ID, TIME])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        corrected = correction_market.join_asof(
            difference_events,
            on=TIME,
            by=["quantile", ASSET_ID],
            strategy="backward",
        )
    return (
        corrected.with_columns(
            (
                pl.col("_weight_difference").fill_null(0.0) * pl.col("forward_return")
            ).alias("_correction")
        )
        .group_by(TIME, "quantile")
        .agg(pl.col("_correction").sum())
    )


def quantile_equal_weights(
    factor: pl.DataFrame,
    *,
    quantiles: int,
) -> dict[str, pl.DataFrame]:
    """Convert factor scores into equal-weight portfolios by quantile."""

    empty = pl.DataFrame(
        schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64}
    )
    bucketed = _quantile_bucket_frame(factor.drop_nulls("factor"), quantiles=quantiles)
    if bucketed.is_empty():
        return {f"q{number}": empty.clone() for number in range(1, quantiles + 1)}
    weights = (
        bucketed.with_columns((1.0 / pl.len().over(TIME, "bucket")).alias("weight"))
        .select(TIME, ASSET_ID, "bucket", "weight")
        .sort([TIME, ASSET_ID])
    )
    return {
        f"q{number}": (
            weights.filter(pl.col("bucket") == number)
            .select(TIME, ASSET_ID, "weight")
            .sort([TIME, ASSET_ID])
            if weights.filter(pl.col("bucket") == number).height
            else empty.clone()
        )
        for number in range(1, quantiles + 1)
    }


def top_n_equal_weights(factor: pl.DataFrame, *, top_n: int) -> pl.DataFrame:
    """Convert factor scores into long-only TOP N equal weights."""

    selected = (
        factor.drop_nulls("factor")
        .sort([TIME, "factor"], descending=[False, True])
        .with_columns(pl.int_range(1, pl.len() + 1).over(TIME).alias("rank"))
        .filter(pl.col("rank") <= top_n)
        .with_columns((1.0 / pl.len().over(TIME)).alias("weight"))
        .select(TIME, ASSET_ID, "weight")
    )
    return selected.sort([TIME, ASSET_ID])


def spread_quantile_weights(
    factor: pl.DataFrame,
    *,
    quantiles: int,
) -> pl.DataFrame:
    """Convert q1 and qN factor buckets into spread portfolio weights."""

    bucketed = _quantile_bucket_frame(factor.drop_nulls("factor"), quantiles=quantiles)
    selected = bucketed.filter(pl.col("bucket").is_in([1, quantiles]))
    if selected.is_empty():
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64}
        )
    return (
        selected.with_columns(
            pl.when(pl.col("bucket") == 1)
            .then(1.0 / pl.len().over(TIME, "bucket"))
            .otherwise(-1.0 / pl.len().over(TIME, "bucket"))
            .alias("weight")
        )
        .select(TIME, ASSET_ID, "weight")
        .sort([TIME, ASSET_ID])
    )


def factor_lag_analysis(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    lags: tuple[int, ...] = FACTOR_LAGS,
) -> pl.DataFrame:
    """Backtest TOP N and spread signals delayed by trading sessions."""

    analysis, _, _ = _lag_outputs(factor, prices, config=config, lags=lags)
    return analysis


def factor_lag_returns(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    lags: tuple[int, ...] = FACTOR_LAGS,
) -> pl.DataFrame:
    """Return cumulative returns for signals delayed by trading sessions."""

    _, returns, _ = _lag_outputs(factor, prices, config=config, lags=lags)
    return returns


def factor_ic_decay(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    trading_sessions: pl.DataFrame | None = None,
    return_provider: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    lags: tuple[int, ...] = FACTOR_LAGS,
) -> pl.DataFrame:
    """Compute IC decay for signals delayed by trading sessions."""

    calendar = (
        _trading_sessions(trading_sessions)
        if trading_sessions is not None
        else _trading_sessions(forward_returns)
    )
    if return_provider is None:
        return _batched_factor_ic_decay(
            factor,
            forward_returns,
            trading_sessions=calendar,
            lags=lags,
        )
    rows: list[dict[str, object]] = []
    for lag in lags:
        lagged = lag_factor(factor, lag=lag, trading_sessions=calendar)
        if lagged.is_empty():
            rows.extend(
                [
                    {"lag": lag, "method": "pearson", "ic_mean": math.nan},
                    {"lag": lag, "method": "spearman", "ic_mean": math.nan},
                ]
            )
            continue
        ic = information_coefficients(
            lagged,
            return_provider(lagged) if return_provider is not None else forward_returns,
        )
        summary = summarize_ic(ic, annualization=1)
        for row in summary.iter_rows(named=True):
            rows.append(
                {
                    "lag": lag,
                    "method": row["method"],
                    "ic_mean": row["mean"],
                }
            )
    return pl.DataFrame(rows).sort(["method", "lag"])


def factor_ic_decay_series(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    trading_sessions: pl.DataFrame | None = None,
    return_provider: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    lags: tuple[int, ...] = FACTOR_LAGS,
) -> pl.DataFrame:
    """Compute per-date Pearson and Spearman IC values for every lag."""

    calendar = (
        _trading_sessions(trading_sessions)
        if trading_sessions is not None
        else _trading_sessions(forward_returns)
    )
    if return_provider is None:
        return _batched_factor_ic_decay_series(
            factor,
            forward_returns,
            trading_sessions=calendar,
            lags=lags,
        )
    frames: list[pl.DataFrame] = []
    for lag in lags:
        lagged = lag_factor(factor, lag=lag, trading_sessions=calendar)
        if lagged.is_empty():
            continue
        ic = information_coefficients(lagged, return_provider(lagged))
        frames.append(
            ic.unpivot(
                index=TIME,
                on=["pearson_ic", "spearman_ic"],
                variable_name="method",
                value_name="ic",
            ).with_columns(
                pl.lit(lag).alias("lag"),
                pl.col("method").str.replace("_ic$", ""),
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "lag": pl.Int64,
                "method": pl.String,
                "ic": pl.Float64,
            }
        )
    return (
        pl.concat(frames)
        .select(TIME, "lag", "method", "ic")
        .sort(["method", "lag", TIME])
    )


def _batched_factor_ic_decay(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    trading_sessions: pl.DataFrame,
    lags: tuple[int, ...],
) -> pl.DataFrame:
    """Map and aggregate every fixed-horizon IC lag in one Polars plan."""

    lag_frame = pl.DataFrame({"lag": lags}, schema={"lag": pl.Int64})
    result_grid = lag_frame.join(
        pl.DataFrame(
            {"method": ["pearson", "spearman"]},
            schema={"method": pl.String},
        ),
        how="cross",
    )
    if factor.is_empty() or not lags:
        return result_grid.with_columns(pl.lit(float("nan")).alias("ic_mean")).sort(
            ["method", "lag"]
        )

    series = _batched_factor_ic_decay_series(
        factor,
        forward_returns,
        trading_sessions=trading_sessions,
        lags=lags,
    )
    aggregated = (
        series.group_by("lag", "method").agg(pl.col("ic").mean().alias("ic_mean"))
        if not series.is_empty()
        else pl.DataFrame(
            schema={"lag": pl.Int64, "method": pl.String, "ic_mean": pl.Float64}
        )
    )
    return (
        result_grid.join(aggregated, on=["lag", "method"], how="left")
        .with_columns(pl.col("ic_mean").fill_null(float("nan")))
        .sort(["method", "lag"])
    )


def _batched_factor_ic_decay_series(
    factor: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    trading_sessions: pl.DataFrame,
    lags: tuple[int, ...],
) -> pl.DataFrame:
    """Map every fixed-horizon IC lag while retaining observation dates."""

    if factor.is_empty() or not lags:
        return pl.DataFrame(
            schema={
                TIME: pl.Date,
                "lag": pl.Int64,
                "method": pl.String,
                "ic": pl.Float64,
            }
        )

    sessions = trading_sessions.with_row_index("_session_index")
    ranked_factor = factor.drop_nulls("factor").with_columns(
        pl.col("factor").rank("average").over(TIME).alias("_source_factor_rank"),
        pl.len().over(TIME).alias("_source_count"),
    )
    lagged_parts: list[pl.DataFrame] = []
    chunk_size = len(lags) if factor.height <= 100_000 else 1
    for start in range(0, len(lags), chunk_size):
        chunk = pl.DataFrame(
            {"lag": lags[start : start + chunk_size]},
            schema={"lag": pl.Int64},
        )
        lag_mapping = (
            sessions.lazy()
            .join(chunk.lazy(), how="cross")
            .select(
                (pl.col("_session_index") - pl.col("lag")).alias("_source_index"),
                pl.col(TIME).alias("_lagged_time"),
                "lag",
            )
        )
        paired = (
            ranked_factor.lazy()
            .join(sessions.lazy(), on=TIME, how="inner")
            .join(
                lag_mapping,
                left_on="_session_index",
                right_on="_source_index",
                how="inner",
            )
            .select(
                pl.col("_lagged_time").alias(TIME),
                ASSET_ID,
                "factor",
                "_source_factor_rank",
                "_source_count",
                "lag",
            )
            .join(forward_returns.lazy(), on=[TIME, ASSET_ID], how="inner")
            .drop_nulls("forward_return")
            .with_columns(
                pl.len().over("lag", TIME).alias("_paired_count"),
            )
            .collect(engine="streaming")
        )
        if paired.is_empty():
            continue
        complete = paired.filter(pl.col("_paired_count") == pl.col("_source_count"))
        incomplete = paired.filter(pl.col("_paired_count") != pl.col("_source_count"))
        if incomplete.height:
            incomplete = incomplete.with_columns(
                pl.col("factor")
                .rank("average")
                .over("lag", TIME)
                .alias("_source_factor_rank")
            )
        paired = (
            pl.concat([complete, incomplete])
            if complete.height and incomplete.height
            else (complete if complete.height else incomplete)
        )
        lagged_parts.append(
            paired.lazy()
            .with_columns(
                pl.col("forward_return")
                .rank("average")
                .over("lag", TIME)
                .alias("_return_rank"),
            )
            .group_by("lag", TIME)
            .agg(
                _corr_expr("factor", "forward_return").alias("pearson"),
                _corr_expr("_source_factor_rank", "_return_rank").alias("spearman"),
            )
            .unpivot(
                index=["lag", TIME],
                on=["pearson", "spearman"],
                variable_name="method",
                value_name="ic",
            )
            .collect(engine="streaming")
        )
    return (
        pl.concat(lagged_parts).sort(["method", "lag", TIME])
        if lagged_parts
        else pl.DataFrame(
            schema={
                TIME: pl.Date,
                "lag": pl.Int64,
                "method": pl.String,
                "ic": pl.Float64,
            }
        )
    )


def lag_factor(
    factor: pl.DataFrame,
    *,
    lag: int,
    trading_sessions: pl.DataFrame,
) -> pl.DataFrame:
    """Move each factor snapshot forward by ``lag`` trading sessions."""

    return _lag_frame(
        factor,
        lag=lag,
        trading_sessions=trading_sessions,
        value_columns=("factor",),
    )


def _lag_frame(
    frame: pl.DataFrame,
    *,
    lag: int,
    trading_sessions: pl.DataFrame,
    value_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Move a keyed snapshot frame forward by trading sessions."""

    if lag <= 0:
        return frame.select(TIME, ASSET_ID, *value_columns).sort([TIME, ASSET_ID])
    sessions = _trading_sessions(trading_sessions).with_row_index("_session_index")
    target_times = sessions.select(
        (pl.col("_session_index") - lag).alias("_session_index"),
        pl.col(TIME).alias("_lagged_time"),
    )
    return (
        frame.join(sessions, on=TIME, how="inner")
        .join(target_times, on="_session_index", how="inner")
        .drop(TIME, "_session_index")
        .rename({"_lagged_time": TIME})
        .select(TIME, ASSET_ID, *value_columns)
        .sort([TIME, ASSET_ID])
    )


def _trading_sessions(frame: pl.DataFrame) -> pl.DataFrame:
    """Return the ordered unique open sessions represented by market data."""

    if TIME not in frame.columns:
        raise InputValidationError(
            f"trading sessions are missing required column: {TIME}"
        )
    return (
        frame.select(pl.col(TIME).cast(pl.Date, strict=False))
        .drop_nulls(TIME)
        .unique()
        .sort(TIME)
    )


def _corr_expr(left: str, right: str) -> pl.Expr:
    return (
        pl.when(
            (pl.len() < 2)
            | (pl.col(left).n_unique() < 2)
            | (pl.col(right).n_unique() < 2)
        )
        .then(None)
        .otherwise(pl.corr(left, right))
    )


def _quantile_bucket_frame(frame: pl.DataFrame, *, quantiles: int) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Int64).alias("bucket")).filter(
            pl.lit(False)
        )
    ranked = (
        frame.sort([TIME, "factor"], descending=[False, True])
        .with_columns(
            pl.len().over(TIME).alias("_count"),
            pl.int_range(1, pl.len() + 1).over(TIME).alias("_rank"),
        )
        .filter(pl.col("_count") >= quantiles)
        .with_columns(
            (((pl.col("_rank") - 1) * quantiles / pl.col("_count")).floor() + 1)
            .cast(pl.Int64)
            .alias("bucket")
        )
        .drop("_count", "_rank")
    )
    return ranked


def _quantile_grid(times: pl.DataFrame, quantiles: int) -> pl.DataFrame:
    quantile_labels = pl.DataFrame(
        {"quantile": [f"q{number}" for number in range(1, quantiles + 1)]},
        schema={"quantile": pl.String},
    )
    return times.select(TIME).unique().join(quantile_labels, how="cross")


def _lag_outputs(
    factor: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    config: BacktestConfig,
    lags: tuple[int, ...],
    forward_returns: pl.DataFrame | None = None,
    price_gaps: pl.DataFrame | None = None,
    execution_availability: pl.DataFrame | None = None,
    lag_zero: Mapping[str, BacktestResult | None] | None = None,
    execution_availability_validated: bool = False,
    market_context: _SparseMarketContext | None = None,
    additional_weight_frames: Mapping[str, pl.DataFrame] | None = None,
    slippage_rates: pl.DataFrame | None = None,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    dict[str, _CompactBacktestResult],
]:
    analysis_rows: list[dict[str, object]] = []
    return_frames: list[pl.DataFrame] = []
    trading_sessions = _trading_sessions(prices)
    resolved_forward_returns = (
        prepare_price_data(prices).forward_returns
        if forward_returns is None
        else forward_returns
    )
    resolved_availability = (
        validate_execution_availability(execution_availability)
        if execution_availability is not None and not execution_availability_validated
        else execution_availability
    )
    portfolio_weights: dict[str, list[tuple[int, pl.DataFrame]]] = {
        "top_n": [],
        "spread": [],
    }
    for lag in lags:
        lagged = lag_factor(factor, lag=lag, trading_sessions=trading_sessions)
        portfolio_weights["top_n"].append(
            (lag, top_n_equal_weights(lagged, top_n=config.top_n))
        )
        portfolio_weights["spread"].append(
            (
                lag,
                spread_quantile_weights(lagged, quantiles=config.quantiles),
            )
        )
    batch_inputs = {
        f"{portfolio}:{lag}": weights
        for portfolio, lagged_weights in portfolio_weights.items()
        for lag, weights in lagged_weights
        if not weights.is_empty()
        and not (lag == 0 and lag_zero and lag_zero.get(portfolio) is not None)
    }
    additional_labels: dict[str, str] = {}
    for label, weights in (additional_weight_frames or {}).items():
        if weights.is_empty():
            continue
        canonical = next(
            (
                candidate
                for candidate, candidate_weights in batch_inputs.items()
                if weights.equals(candidate_weights)
            ),
            None,
        )
        if canonical is None:
            canonical = f"_additional:{label}"
            batch_inputs[canonical] = weights
        additional_labels[label] = canonical
    execution_keys = (
        market_context.execution_keys
        if market_context is not None
        else (
            resolved_forward_returns.select(TIME, ASSET_ID)
            .unique()
            .sort([ASSET_ID, TIME])
        )
    )
    batch_results = (
        _run_sparse_compact_backtests(
            batch_inputs,
            prices,
            resolved_forward_returns,
            config=config,
            execution_availability=resolved_availability,
            execution_availability_validated=True,
            execution_keys=execution_keys,
            market_context=market_context,
            slippage_rates=slippage_rates,
        )
        if batch_inputs
        else {}
    )
    for portfolio, lagged_weights in portfolio_weights.items():
        for lag, weights in lagged_weights:
            backtest = lag_zero.get(portfolio) if lag == 0 and lag_zero else None
            if backtest is None and weights.height:
                backtest = batch_results.get(f"{portfolio}:{lag}")
            row: dict[str, object] = {
                "lag": lag,
                "portfolio": portfolio,
                "gross_cumulative_return": math.nan,
                "net_cumulative_return": math.nan,
                "gross_sharpe": math.nan,
                "net_sharpe": math.nan,
                "is_bankrupt": False,
                "bankruptcy_time": None,
            }
            if backtest is not None:
                bankruptcy_time = (
                    backtest.returns.filter(pl.col("bankruptcy_event"))
                    .get_column(TIME)
                    .min()
                )
                row.update(
                    {
                        "gross_cumulative_return": backtest.summary.gross_total_return,
                        "net_cumulative_return": backtest.summary.net_total_return,
                        "gross_sharpe": backtest.summary.gross_sharpe,
                        "net_sharpe": backtest.summary.net_sharpe,
                        "is_bankrupt": bankruptcy_time is not None,
                        "bankruptcy_time": bankruptcy_time,
                    }
                )
                return_frames.append(
                    backtest.returns.select(
                        pl.lit(lag).alias("lag"),
                        pl.lit(portfolio).alias("portfolio"),
                        TIME,
                        "gross_return",
                        "net_return",
                        "is_bankrupt",
                        "bankruptcy_event",
                    ).join(
                        backtest.value.select(
                            TIME,
                            pl.col("gross_return_cumulative").alias(
                                "gross_cumulative_return"
                            ),
                            pl.col("net_return_cumulative").alias(
                                "net_cumulative_return"
                            ),
                            pl.lit(backtest.summary.gross_sharpe).alias("gross_sharpe"),
                            pl.lit(backtest.summary.net_sharpe).alias("net_sharpe"),
                        ),
                        on=TIME,
                        how="left",
                    )
                )
            analysis_rows.append(row)
    analysis = pl.DataFrame(analysis_rows).sort(["portfolio", "lag"])
    if not return_frames:
        returns = pl.DataFrame(
            schema={
                "lag": pl.Int64,
                "portfolio": pl.String,
                TIME: pl.Date,
                "gross_return": pl.Float64,
                "net_return": pl.Float64,
                "gross_cumulative_return": pl.Float64,
                "net_cumulative_return": pl.Float64,
                "gross_sharpe": pl.Float64,
                "net_sharpe": pl.Float64,
                "is_bankrupt": pl.Boolean,
                "bankruptcy_event": pl.Boolean,
            }
        )
    else:
        returns = pl.concat(return_frames).sort(["portfolio", "lag", TIME])
    return (
        analysis,
        returns,
        {
            label: batch_results[canonical]
            for label, canonical in additional_labels.items()
        },
    )


def _spread_returns(quantile_returns: pl.DataFrame, quantiles: int) -> pl.DataFrame:
    top = "q1"
    bottom = f"q{quantiles}"
    return (
        quantile_returns.filter(pl.col("quantile").is_in([bottom, top]))
        .select(TIME, "quantile", "return")
        .pivot(index=TIME, on="quantile", values="return")
        .with_columns((pl.col(top) - pl.col(bottom)).alias("spread_return"))
        .select(TIME, "spread_return")
        .sort(TIME)
    )


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks
