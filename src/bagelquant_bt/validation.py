"""Validation-only scoring for machine-learning candidate predictions.

The functions in this module are deliberately model-agnostic.  They consume
frozen candidate predictions and targets, or already simulated portfolio
returns, and never fit, mutate, or select a Core model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt

import numpy as np
import polars as pl

TIME = "time"
ASSET_ID = "asset_id"


class ValidationObjective(StrEnum):
    """Supported candidate-selection objectives."""

    MEAN_IC = "mean_ic"
    ICIR = "icir"
    TOP_N_NET_SHARPE = "top_n_net_sharpe"
    TOP_N_NET_SHARPE_WITH_TURNOVER_REGULARIZATION = (
        "top_n_net_sharpe_with_turnover_regularization"
    )


@dataclass(frozen=True, slots=True)
class MonthlyIcObservation:
    """Auditable validity and value for one validation month."""

    time: object
    ic: float | None
    valid: bool
    reason: str | None
    pair_count: int


@dataclass(frozen=True, slots=True)
class ValidationScore:
    """Candidate score plus its minimum-month gate."""

    objective: ValidationObjective
    score: float | None
    valid_months: int
    required_months: int
    valid: bool
    reason: str | None
    monthly_ic: tuple[MonthlyIcObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class TopNSelectionResult:
    """Stable exact-N selections and cutoff-tie audit rows."""

    selections: pl.DataFrame
    audit: pl.DataFrame


def score_ic_validation(
    predictions: pl.DataFrame,
    *,
    minimum_valid_months: int,
    objective: str | ValidationObjective = ValidationObjective.MEAN_IC,
    minimum_pairs: int = 2,
    annualization: int = 12,
    expected_periods: Sequence[object] | None = None,
) -> ValidationScore:
    """Score monthly cross-sectional Spearman IC with explicit invalid months."""

    resolved = ValidationObjective(objective)
    if resolved not in {ValidationObjective.MEAN_IC, ValidationObjective.ICIR}:
        raise ValueError("score_ic_validation requires mean_ic or icir")
    if minimum_valid_months <= 0:
        raise ValueError("minimum_valid_months must be positive")
    if minimum_pairs < 2:
        raise ValueError("minimum_pairs must be at least two")
    _require_columns(predictions, {TIME, ASSET_ID, "prediction", "target"})
    observations: list[MonthlyIcObservation] = []
    periods = (
        sorted(set(expected_periods))
        if expected_periods is not None
        else predictions.get_column(TIME).unique().sort().to_list()
    )
    for period in periods:
        current = predictions.filter(pl.col(TIME) == period).select(
            ASSET_ID,
            pl.col("prediction").cast(pl.Float64, strict=False),
            pl.col("target").cast(pl.Float64, strict=False),
        )
        pairs = current.filter(
            pl.col("prediction").is_finite() & pl.col("target").is_finite()
        )
        reason: str | None = None
        if pairs.height < minimum_pairs:
            reason = "insufficient_pairs"
        elif pairs.get_column("prediction").n_unique() < 2:
            reason = "constant_prediction"
        elif pairs.get_column("target").n_unique() < 2:
            reason = "constant_target"
        if reason is not None:
            observations.append(
                MonthlyIcObservation(period, None, False, reason, pairs.height)
            )
            continue
        prediction_rank = pairs.get_column("prediction").rank("average").to_numpy()
        target_rank = pairs.get_column("target").rank("average").to_numpy()
        ic = float(np.corrcoef(prediction_rank, target_rank)[0, 1])
        if not np.isfinite(ic):
            observations.append(
                MonthlyIcObservation(
                    period, None, False, "undefined_correlation", pairs.height
                )
            )
        else:
            observations.append(
                MonthlyIcObservation(period, ic, True, None, pairs.height)
            )
    valid_values = np.asarray(
        [item.ic for item in observations if item.valid], dtype=float
    )
    if valid_values.size < minimum_valid_months:
        return ValidationScore(
            objective=resolved,
            score=None,
            valid_months=int(valid_values.size),
            required_months=minimum_valid_months,
            valid=False,
            reason="insufficient_valid_validation_months",
            monthly_ic=tuple(observations),
        )
    mean = float(valid_values.mean())
    if resolved == ValidationObjective.MEAN_IC:
        score = mean
    else:
        standard_deviation = (
            float(valid_values.std(ddof=1)) if valid_values.size > 1 else np.nan
        )
        score = (
            mean / standard_deviation * sqrt(annualization)
            if np.isfinite(standard_deviation) and standard_deviation > 0
            else 0.0
        )
    return ValidationScore(
        objective=resolved,
        score=score,
        valid_months=int(valid_values.size),
        required_months=minimum_valid_months,
        valid=True,
        reason=None,
        monthly_ic=tuple(observations),
    )


def select_top_n_stable(
    predictions: pl.DataFrame,
    *,
    top_n: int,
    expected_periods: Sequence[object] | None = None,
) -> TopNSelectionResult:
    """Select exactly N finite predictions using asset id as the tie-breaker."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    _require_columns(predictions, {TIME, ASSET_ID, "prediction"})
    selections: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    periods = (
        sorted(set(expected_periods))
        if expected_periods is not None
        else predictions.get_column(TIME).unique().sort().to_list()
    )
    for period in periods:
        current = (
            predictions.filter(pl.col(TIME) == period)
            .select(
                TIME,
                pl.col(ASSET_ID).cast(pl.String),
                pl.col("prediction").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("prediction").is_finite())
            .sort(["prediction", ASSET_ID], descending=[True, False])
        )
        reason: str | None = None
        if current.height < top_n:
            reason = "underfilled_top_n"
        elif current.get_column("prediction").n_unique() < 2:
            reason = "constant_prediction"
        if reason is not None:
            audit.append(
                {
                    TIME: period,
                    "valid": False,
                    "reason": reason,
                    "eligible_count": current.height,
                    "cutoff_score": None,
                    "cutoff_tie_count": 0,
                    "tie_excluded_count": 0,
                }
            )
            continue
        selected = current.head(top_n)
        cutoff = float(selected.get_column("prediction")[-1])
        tie_count = current.filter(pl.col("prediction") == cutoff).height
        selected_ties = selected.filter(pl.col("prediction") == cutoff).height
        for row in selected.iter_rows(named=True):
            selections.append(
                {
                    TIME: row[TIME],
                    ASSET_ID: row[ASSET_ID],
                    "prediction": row["prediction"],
                    "weight": 1.0 / top_n,
                }
            )
        audit.append(
            {
                TIME: period,
                "valid": True,
                "reason": None,
                "eligible_count": current.height,
                "cutoff_score": cutoff,
                "cutoff_tie_count": tie_count,
                "tie_excluded_count": tie_count - selected_ties,
            }
        )
    selection_schema = {
        TIME: predictions.schema[TIME],
        ASSET_ID: pl.String,
        "prediction": pl.Float64,
        "weight": pl.Float64,
    }
    audit_schema = {
        TIME: predictions.schema[TIME],
        "valid": pl.Boolean,
        "reason": pl.String,
        "eligible_count": pl.Int64,
        "cutoff_score": pl.Float64,
        "cutoff_tie_count": pl.Int64,
        "tie_excluded_count": pl.Int64,
    }
    return TopNSelectionResult(
        pl.DataFrame(selections, schema=selection_schema).sort([TIME, ASSET_ID]),
        pl.DataFrame(audit, schema=audit_schema).sort(TIME),
    )


def score_top_n_performance(
    monthly_performance: pl.DataFrame,
    *,
    minimum_valid_months: int,
    objective: str | ValidationObjective,
    turnover_penalty: float | None = None,
    annualization: int = 12,
) -> ValidationScore:
    """Score BT-produced net returns, optionally regularized by turnover."""

    resolved = ValidationObjective(objective)
    allowed = {
        ValidationObjective.TOP_N_NET_SHARPE,
        ValidationObjective.TOP_N_NET_SHARPE_WITH_TURNOVER_REGULARIZATION,
    }
    if resolved not in allowed:
        raise ValueError("score_top_n_performance requires a Top-N objective")
    if minimum_valid_months <= 0:
        raise ValueError("minimum_valid_months must be positive")
    required_columns = {TIME, "net_return"}
    if resolved == ValidationObjective.TOP_N_NET_SHARPE_WITH_TURNOVER_REGULARIZATION:
        required_columns.add("turnover")
        if turnover_penalty is None or not np.isfinite(turnover_penalty):
            raise ValueError("turnover_penalty must be explicitly finite")
        if turnover_penalty < 0:
            raise ValueError("turnover_penalty cannot be negative")
    _require_columns(monthly_performance, required_columns)
    valid = monthly_performance.filter(pl.col("net_return").is_finite())
    if valid.height < minimum_valid_months:
        return ValidationScore(
            objective=resolved,
            score=None,
            valid_months=valid.height,
            required_months=minimum_valid_months,
            valid=False,
            reason="insufficient_valid_validation_months",
        )
    returns = valid.get_column("net_return").to_numpy()
    deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
    net_sharpe = (
        float(np.mean(returns) / deviation * sqrt(annualization))
        if np.isfinite(deviation) and deviation > 0
        else 0.0
    )
    score = net_sharpe
    if resolved == ValidationObjective.TOP_N_NET_SHARPE_WITH_TURNOVER_REGULARIZATION:
        turnovers = valid.get_column("turnover").cast(pl.Float64, strict=False)
        if turnovers.null_count() or not turnovers.is_finite().all():
            return ValidationScore(
                objective=resolved,
                score=None,
                valid_months=valid.height,
                required_months=minimum_valid_months,
                valid=False,
                reason="invalid_turnover",
            )
        score -= float(turnover_penalty) * float(turnovers.mean())
    return ValidationScore(
        objective=resolved,
        score=score,
        valid_months=valid.height,
        required_months=minimum_valid_months,
        valid=True,
        reason=None,
    )


def top_n_monthly_performance(
    predictions: pl.DataFrame,
    *,
    top_n: int,
    cost_rate_per_turnover: float = 0.0,
    expected_periods: Sequence[object] | None = None,
    commission_rate: float | None = None,
    minimum_fee: float = 0.0,
    sell_tax_rate: float = 0.0,
    slippage_rate: float = 0.0,
    initial_capital: float = 1.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build exact-N equal-weight monthly returns for candidate validation.

    ``forward_return`` is the realized execution-to-next-execution return.
    Costs are applied to canonical absolute weight turnover.  This compact
    evaluator is intended for candidate selection; production simulations may
    use the full engine for richer fee schedules.
    """

    if not np.isfinite(cost_rate_per_turnover) or cost_rate_per_turnover < 0:
        raise ValueError("cost_rate_per_turnover must be finite and non-negative")
    detailed_costs = commission_rate is not None
    if detailed_costs and cost_rate_per_turnover:
        raise ValueError(
            "cost_rate_per_turnover cannot be combined with detailed costs"
        )
    for name, value in {
        "minimum_fee": minimum_fee,
        "sell_tax_rate": sell_tax_rate,
        "slippage_rate": slippage_rate,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if detailed_costs and (
        not np.isfinite(commission_rate) or float(commission_rate) < 0
    ):
        raise ValueError("commission_rate must be finite and non-negative")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be positive and finite")
    _require_columns(predictions, {TIME, ASSET_ID, "prediction", "forward_return"})
    selection = select_top_n_stable(
        predictions, top_n=top_n, expected_periods=expected_periods
    )
    selected = selection.selections.join(
        predictions.select(TIME, ASSET_ID, "forward_return"),
        on=[TIME, ASSET_ID],
        how="inner",
    ).filter(pl.col("forward_return").cast(pl.Float64, strict=False).is_finite())
    rows: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    equity = float(initial_capital)
    valid_periods = set(
        selection.audit.filter(pl.col("valid")).get_column(TIME).to_list()
    )
    for period in selection.audit.get_column(TIME).to_list():
        if period not in valid_periods:
            continue
        current = selected.filter(pl.col(TIME) == period)
        if current.height != top_n:
            continue
        weights = {
            str(row[ASSET_ID]): float(row["weight"])
            for row in current.iter_rows(named=True)
        }
        assets = set(previous) | set(weights)
        turnover = sum(
            abs(weights.get(asset, 0.0) - previous.get(asset, 0.0))
            for asset in assets
        )
        gross_return = float(
            (current.get_column("weight") * current.get_column("forward_return")).sum()
        )
        if detailed_costs:
            transaction_cost = 0.0
            for asset in assets:
                change = weights.get(asset, 0.0) - previous.get(asset, 0.0)
                notional = abs(change) * equity
                if notional <= 0:
                    continue
                transaction_cost += max(
                    float(commission_rate) * notional, minimum_fee
                )
                transaction_cost += slippage_rate * notional
                if change < 0:
                    transaction_cost += sell_tax_rate * notional
            cost_return = transaction_cost / equity
        else:
            cost_return = cost_rate_per_turnover * turnover
        net_return = gross_return - cost_return
        rows.append(
            {
                TIME: period,
                "gross_return": gross_return,
                "turnover": turnover,
                "net_return": net_return,
            }
        )
        equity *= 1.0 + net_return
        previous = weights
    schema = {
        TIME: predictions.schema[TIME],
        "gross_return": pl.Float64,
        "turnover": pl.Float64,
        "net_return": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema).sort(TIME), selection.audit


def _require_columns(frame: pl.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


__all__ = [
    "MonthlyIcObservation",
    "TopNSelectionResult",
    "ValidationObjective",
    "ValidationScore",
    "score_ic_validation",
    "score_top_n_performance",
    "select_top_n_stable",
    "top_n_monthly_performance",
]
