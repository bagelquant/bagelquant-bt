"""Strict AlphaValue-to-Prediction-to-Weights backtest orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import polars as pl
from bagelquant_core import (
    Panel,
    PredictionComposer,
    PredictionPanel,
    PredictionTrainingContext,
)

from .config import BacktestConfig
from .engine import run_weight_backtest
from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME, validate_prices
from .policy import (
    AlphaPolicy,
    AlphaPolicyResult,
    ExecutionPolicy,
    resolve_execution_policy,
)
from .results import BacktestResult


def compose_prediction(
    alpha_values: Mapping[str, Panel],
    composer: PredictionComposer,
    calendar: pl.DataFrame,
    alpha_policy: AlphaPolicy,
    *,
    execution_policy: ExecutionPolicy | str = "next_open",
    prices: pl.DataFrame | None = None,
    start: date | None = None,
    end: date | None = None,
) -> PredictionPanel:
    """Apply Alpha Policy, then return the Composer's raw PredictionPanel."""

    if not isinstance(alpha_policy, AlphaPolicy):
        raise TypeError("alpha_policy must be an AlphaPolicy")
    processed = alpha_policy.apply(
        alpha_values,
        calendar,
        start=start,
        end=end,
    )
    return compose_processed_prediction(
        processed,
        composer,
        calendar,
        execution_policy=execution_policy,
        prices=prices,
    )


def compose_processed_prediction(
    processed: AlphaPolicyResult,
    composer: PredictionComposer,
    calendar: pl.DataFrame,
    *,
    execution_policy: ExecutionPolicy | str = "next_open",
    prices: pl.DataFrame | None = None,
) -> PredictionPanel:
    """Compose AlphaPolicy-processed Panels without applying the policy again."""

    if not isinstance(processed, AlphaPolicyResult):
        raise TypeError("processed must be an AlphaPolicyResult")
    if not processed.alpha_values:
        raise ValueError("processed AlphaPolicy result contains no AlphaValues")
    if not isinstance(composer, PredictionComposer):
        raise TypeError("composer must be a PredictionComposer")
    panels = tuple(processed.alpha_values.values())
    if any(not isinstance(panel, Panel) for panel in panels):
        raise TypeError("processed alpha_values must contain Panel values")
    policy_ids = {str(panel.metadata.get("alpha_policy", "")) for panel in panels}
    standardizations = {
        str(panel.metadata.get("standardization", "")) for panel in panels
    }
    if len(policy_ids) != 1 or "" in policy_ids:
        raise ValueError("processed AlphaValues must share one Alpha Policy")
    if len(standardizations) != 1 or "" in standardizations:
        raise ValueError("processed AlphaValues must share one standardization")
    training = None
    if composer.supervised:
        if prices is None:
            raise InputValidationError(
                f"{composer.kind} prediction composition requires execution prices"
            )
        training = _training_context(
            processed.alpha_values,
            processed.schedule,
            calendar,
            execution_policy,
            prices,
        )
    graph = composer.compose(
        *panels,
        training=training,
        name="prediction",
        metadata={
            "prediction_composer": composer.kind,
            "window": composer.window,
            "alpha_policy": next(iter(policy_ids)),
            "standardization": next(iter(standardizations)),
        },
    )
    result = graph.compute(dense_output=False)
    if not isinstance(result, PredictionPanel):
        raise AssertionError("prediction composer did not produce a PredictionPanel")
    return result


def run_prediction_backtest(
    prediction: PredictionPanel,
    prices: pl.DataFrame,
    calendar: pl.DataFrame,
    *,
    weight_policy: object,
    execution_policy: ExecutionPolicy | str = "next_open",
    weight_inputs: Mapping[str, object] | None = None,
    config: BacktestConfig | None = None,
    execution_availability: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
) -> BacktestResult:
    """Build evaluation-date weights, schedule them, then run the engine."""

    if not isinstance(prediction, PredictionPanel):
        raise TypeError("run_prediction_backtest requires a PredictionPanel")
    if config is None:
        raise ValueError("config is required")
    build = getattr(weight_policy, "build", None)
    if build is None:
        raise TypeError("weight_policy must define build(PredictionPanel, ...)")
    result = build(
        prediction,
        prices=prices,
        config=config,
        **dict(weight_inputs or {}),
    )
    if not isinstance(result.weights, Panel) or isinstance(
        result.weights, PredictionPanel
    ):
        raise TypeError("WeightPolicy must return an ordinary weights Panel")
    execution = (
        resolve_execution_policy(execution_policy)
        if isinstance(execution_policy, str)
        else execution_policy
    )
    scheduled = execution.schedule_weights(result.weights, calendar)
    weights = (
        scheduled.weights.collect(dense=False)
        .drop_nulls("value")
        .rename({"value": "weight"})
    )
    return run_weight_backtest(
        weights,
        prices,
        config=config,
        execution_availability=execution_availability,
        slippage_rates=slippage_rates,
    )


def _training_context(
    alpha_values: Mapping[str, Panel],
    schedule: pl.DataFrame,
    calendar: pl.DataFrame,
    execution_policy: ExecutionPolicy | str,
    prices: pl.DataFrame,
) -> PredictionTrainingContext:
    execution = (
        resolve_execution_policy(execution_policy)
        if isinstance(execution_policy, str)
        else execution_policy
    )
    executable = execution.resolve(schedule, calendar).drop_nulls("execution_date")
    pairs = (
        executable.select("rebalance_date", "execution_date")
        .sort("rebalance_date")
        .with_columns(pl.col("execution_date").shift(-1).alias("next_execution_date"))
        .drop_nulls("next_execution_date")
    )
    aligned_prices = validate_prices(prices)
    targets = (
        pairs.join(
            aligned_prices.rename({TIME: "execution_date", "price": "start_price"}),
            on="execution_date",
            how="inner",
        )
        .join(
            aligned_prices.select(
                pl.col(TIME).alias("next_execution_date"),
                ASSET_ID,
                pl.col("price").alias("end_price"),
            ),
            on=["next_execution_date", ASSET_ID],
            how="inner",
        )
        .select(
            pl.col("rebalance_date").alias(TIME),
            ASSET_ID,
            (pl.col("end_price") / pl.col("start_price") - 1.0).alias("value"),
            "next_execution_date",
        )
        .sort([TIME, ASSET_ID])
    )
    availability = targets.select(
        TIME,
        ASSET_ID,
        pl.col("next_execution_date")
        .dt.epoch("d")
        .cast(pl.Float64)
        .add(date(1970, 1, 1).toordinal())
        .alias("value"),
    )
    domain = next(iter(alpha_values.values())).domain
    return PredictionTrainingContext(
        Panel.from_domain(
            targets.select(TIME, ASSET_ID, "value"),
            domain,
            name="forward_return",
        ),
        Panel.from_domain(availability, domain, name="label_availability"),
    )


__all__ = [
    "compose_prediction",
    "compose_processed_prediction",
    "run_prediction_backtest",
]
