"""Strict AlphaValue-to-Signal-to-Weights backtest orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import polars as pl
from bagelquant_core import (
    Domain,
    Panel,
    SignalComposer,
    SignalPanel,
    SignalStandardization,
    SignalTrainingContext,
)

from .config import BacktestConfig
from .engine import run_weight_backtest
from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME, validate_prices
from .portfolio import EqualWeightPolicy
from .results import BacktestResult
from .signal import ExecutionPolicy, SignalDatePolicy


def compose_signal(
    alpha_values: Mapping[str, Panel],
    composer: SignalComposer,
    calendar: pl.DataFrame,
    signal_date_policy: SignalDatePolicy,
    *,
    standardization: str | SignalStandardization = SignalStandardization.ZSCORE,
    execution_policy: ExecutionPolicy | str = "next_open",
    prices: pl.DataFrame | None = None,
    start: date | None = None,
    end: date | None = None,
) -> SignalPanel:
    """Compose scheduled AlphaValue snapshots into an observation-time signal."""

    if not alpha_values:
        raise ValueError("compose_signal requires at least one AlphaValue Panel")
    if not isinstance(signal_date_policy, SignalDatePolicy):
        raise TypeError("signal_date_policy must be a SignalDatePolicy")
    if any(
        not isinstance(value, Panel) or isinstance(value, SignalPanel)
        for value in alpha_values.values()
    ):
        raise TypeError("alpha_values must contain ordinary Panel values")
    domains = [value.domain for value in alpha_values.values()]
    if any(not domains[0].equivalent_to(domain) for domain in domains[1:]):
        raise ValueError("AlphaValue Panels must use equivalent Domains")

    schedule = signal_date_policy.schedule(calendar, start=start, end=end)
    if schedule.is_empty():
        raise InputValidationError("signal date policy resolves no observations")
    scheduled_values = {
        name: _scheduled_alpha_panel(value, schedule, signal_date_policy)
        for name, value in alpha_values.items()
    }
    training = None
    if composer.supervised:
        if prices is None:
            raise InputValidationError(
                f"{composer.kind} signal composition requires execution prices"
            )
        training = _training_context(
            scheduled_values,
            schedule,
            calendar,
            execution_policy,
            prices,
        )
    graph = composer.compose(
        *scheduled_values.values(),
        standardization=standardization,
        training=training,
        name="signal",
        metadata={
            "composer": composer.kind,
            "window": composer.window,
            "standardization": SignalStandardization(standardization).value,
            "signal_date_policy": signal_date_policy.id,
        },
    )
    result = graph.compute(dense_output=False)
    if not isinstance(result, SignalPanel):
        raise AssertionError("signal composer did not produce a SignalPanel")
    return result


def run_signal_backtest(
    signal: SignalPanel,
    prices: pl.DataFrame,
    calendar: pl.DataFrame,
    signal_date_policy: SignalDatePolicy,
    *,
    execution_policy: ExecutionPolicy | str = "next_open",
    portfolio_policy: object | None = None,
    portfolio_inputs: Mapping[str, object] | None = None,
    config: BacktestConfig | None = None,
    execution_availability: pl.DataFrame | None = None,
    slippage_rates: pl.DataFrame | None = None,
) -> BacktestResult:
    """Schedule a SignalPanel, build target weights, and run the private engine."""

    if not isinstance(signal, SignalPanel):
        raise TypeError("run_signal_backtest requires a SignalPanel")
    if config is None:
        raise ValueError("config is required")
    scheduled = signal_date_policy.select(
        signal,
        calendar,
        execution_policy=execution_policy,
    )
    selected = portfolio_policy or EqualWeightPolicy(config.top_n)
    build = getattr(selected, "build", None)
    if build is None:
        raise TypeError("portfolio_policy must define build(ScheduledSignal, ...)")
    portfolio = build(
        scheduled,
        prices=prices,
        config=config,
        **dict(portfolio_inputs or {}),
    )
    if not isinstance(portfolio.weights, Panel) or isinstance(
        portfolio.weights, SignalPanel
    ):
        raise TypeError("PortfolioPolicy must return an ordinary weights Panel")
    weights = (
        portfolio.weights.collect(dense=False)
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


def _scheduled_alpha_panel(
    alpha: Panel,
    schedule: pl.DataFrame,
    policy: SignalDatePolicy,
) -> Panel:
    values = alpha.collect(dense=False).drop_nulls("value")
    available = values.select(TIME).unique().sort(TIME).get_column(TIME)
    frames: list[pl.DataFrame] = []
    for row in schedule.iter_rows(named=True):
        rebalance = row["rebalance_date"]
        candidates = available.filter(available <= rebalance)
        if policy.frequency == "monthly":
            candidates = candidates.filter(
                candidates.dt.strftime("%Y-%m") == row["period"]
            )
        selected = (
            rebalance
            if rebalance in candidates
            else candidates.max()
            if policy.missing_snapshot.value == "previous_in_period"
            and not candidates.is_empty()
            else None
        )
        if selected is None:
            continue
        frames.append(
            values.filter(pl.col(TIME) == selected).with_columns(
                pl.lit(rebalance, dtype=pl.Date).alias(TIME)
            )
        )
    selected_values = (
        pl.concat(frames).sort([TIME, ASSET_ID])
        if frames
        else pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, "value": pl.Float64}
        )
    )
    domain = Domain(
        calendar=schedule.get_column("rebalance_date"),
        universe=alpha.domain.asset_ids,
    )
    return Panel.from_domain(selected_values, domain, name=alpha.name)


def _training_context(
    alpha_values: Mapping[str, Panel],
    schedule: pl.DataFrame,
    calendar: pl.DataFrame,
    execution_policy: ExecutionPolicy | str,
    prices: pl.DataFrame,
) -> SignalTrainingContext:
    from .signal import resolve_execution_policy

    execution = (
        resolve_execution_policy(execution_policy)
        if isinstance(execution_policy, str)
        else execution_policy
    )
    executable = execution.resolve(schedule, calendar).drop_nulls("execution_date")
    pairs = (
        executable.select("rebalance_date", "execution_date")
        .sort("rebalance_date")
        .with_columns(
            pl.col("execution_date").shift(-1).alias("next_execution_date")
        )
        .drop_nulls("next_execution_date")
    )
    aligned_prices = validate_prices(prices)
    targets = (
        pairs.join(
            aligned_prices.rename(
                {TIME: "execution_date", "price": "start_price"}
            ),
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
    return SignalTrainingContext(
        Panel.from_domain(
            targets.select(TIME, ASSET_ID, "value"),
            domain,
            name="forward_return",
        ),
        Panel.from_domain(availability, domain, name="label_availability"),
    )


__all__ = ["compose_signal", "run_signal_backtest"]
