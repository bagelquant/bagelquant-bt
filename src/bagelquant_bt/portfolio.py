"""Deterministic prediction-to-weight policies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from bagelquant_core import Panel, PredictionPanel

from .engine import backtest_weight_frame
from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME, validate_panel_frame


@dataclass(frozen=True, slots=True)
class WeightBuild:
    weights: Panel
    skipped: pl.DataFrame
    diagnostics: pl.DataFrame = field(default_factory=pl.DataFrame)


@dataclass(frozen=True, slots=True)
class EqualWeightPolicy:
    top_n: int

    def build(self, prediction: PredictionPanel, **_: object) -> WeightBuild:
        selected = _top_n(prediction, self.top_n)
        return WeightBuild(
            _weight_panel(_normalise(selected, "_unit"), prediction),
            _empty_skipped(),
        )


@dataclass(frozen=True, slots=True)
class FloatMarketCapWeightPolicy:
    top_n: int
    market_cap_column: str = "float_market_cap"

    def build(
        self,
        prediction: PredictionPanel,
        *,
        market_caps: pl.DataFrame | None = None,
        **_: object,
    ) -> WeightBuild:
        if market_caps is None:
            raise InputValidationError("float market-cap policy requires market_caps")
        selected = _top_n(prediction, self.top_n)
        caps = validate_panel_frame(
            market_caps, label="market_caps", value_columns=(self.market_cap_column,)
        )
        weighted = selected.join(caps, on=[TIME, ASSET_ID], how="left")
        missing = weighted.filter(pl.col(self.market_cap_column).is_null())
        if missing.height:
            dates = ", ".join(
                str(value) for value in missing.get_column(TIME).unique().sort()
            )
            raise InputValidationError(
                f"missing float market cap for selected predictions at: {dates}"
            )
        invalid = weighted.filter(
            ~pl.col(self.market_cap_column).is_finite()
            | (pl.col(self.market_cap_column) <= 0)
        )
        if invalid.height:
            raise InputValidationError("float market caps must be finite and positive")
        return WeightBuild(
            _weight_panel(_normalise(weighted, self.market_cap_column), prediction),
            _empty_skipped(),
        )


@dataclass(frozen=True, slots=True)
class TargetVolatilityPolicy:
    base: EqualWeightPolicy | FloatMarketCapWeightPolicy
    target_annual_volatility: float = 0.15
    lookback_sessions: int = 60
    annualization: int = 240
    max_gross_exposure: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.target_annual_volatility <= 0
            or self.lookback_sessions <= 1
            or self.annualization <= 0
        ):
            raise ValueError("target volatility settings must be positive")
        if not 0 < self.max_gross_exposure <= 1.0:
            raise ValueError("max_gross_exposure must be in (0, 1]")

    def build(
        self,
        prediction: PredictionPanel,
        *,
        prices: pl.DataFrame | None = None,
        config=None,
        **kwargs: object,
    ) -> WeightBuild:
        if prices is None or config is None:
            raise InputValidationError(
                "target-volatility policy requires prices and config"
            )
        base_panel = self.base.build(prediction, **kwargs).weights
        base = _weight_frame(base_panel)
        history = backtest_weight_frame(base, prices, config=config).returns
        dates = base.select(TIME).unique().sort(TIME)
        scales = []
        for value in dates.get_column(TIME):
            sample = history.filter(pl.col(TIME) < value).tail(self.lookback_sessions)
            if sample.height < self.lookback_sessions:
                scales.append(
                    {
                        TIME: value,
                        "scale": None,
                        "reason": "insufficient_volatility_history",
                    }
                )
                continue
            volatility = (
                float(sample.get_column("gross_return").std()) * self.annualization**0.5
            )
            scale = (
                self.max_gross_exposure
                if volatility == 0
                else min(
                    self.max_gross_exposure, self.target_annual_volatility / volatility
                )
            )
            scales.append({TIME: value, "scale": scale, "reason": None})
        scale_frame = pl.DataFrame(scales)
        weights = (
            base.join(scale_frame.select(TIME, "scale"), on=TIME, how="left")
            .drop_nulls("scale")
            .with_columns((pl.col("weight") * pl.col("scale")).alias("weight"))
            .select(TIME, ASSET_ID, "weight")
        )
        skipped = scale_frame.filter(pl.col("scale").is_null()).select(TIME, "reason")
        return WeightBuild(
            Panel.from_domain(
                weights.rename({"weight": "value"}),
                prediction.domain,
                name="weights",
            ),
            skipped,
        )


@dataclass(frozen=True, slots=True)
class PredictionRegularizedOptimizerPolicy:
    """Long-only prediction optimizer with concentration and turnover penalties."""

    concentration_penalty: float
    turnover_penalty: float
    max_weight: float
    constraint_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        values = (
            self.concentration_penalty,
            self.turnover_penalty,
            self.max_weight,
            self.constraint_tolerance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("optimizer settings must be finite")
        if self.concentration_penalty <= 0:
            raise ValueError("concentration_penalty must be positive")
        if self.turnover_penalty < 0:
            raise ValueError("turnover_penalty must be nonnegative")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if self.constraint_tolerance <= 0:
            raise ValueError("constraint_tolerance must be positive")

    def build(
        self,
        prediction: PredictionPanel,
        *,
        reference_weights: Panel | pl.DataFrame | None = None,
        **_: object,
    ) -> WeightBuild:
        """Optimize each evaluation date against its actual reference holdings."""

        if not isinstance(prediction, PredictionPanel):
            raise TypeError("weight policies require a PredictionPanel")
        predictions = prediction.collect(dense=True).rename({"value": "prediction"})
        references = _reference_weight_frame(reference_weights)
        try:
            import cvxpy as cp
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise InputValidationError(
                "prediction_regularized_optimizer requires bagelquant-bt[optimizer]"
            ) from exc

        policy_hash = _optimizer_policy_hash(self)
        weight_rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for evaluation_date in predictions.get_column(TIME).unique().sort():
            cross_section = predictions.filter(pl.col(TIME) == evaluation_date)
            valid = cross_section.filter(
                pl.col("prediction").is_not_null() & pl.col("prediction").is_finite()
            ).sort(ASSET_ID)
            valid_count = valid.height
            if valid_count == 0:
                raise InputValidationError(
                    f"optimizer has no finite predictions at {evaluation_date}"
                )
            if valid_count * self.max_weight < 1.0 - self.constraint_tolerance:
                raise InputValidationError(
                    "optimizer constraints are infeasible at "
                    f"{evaluation_date}: {valid_count} valid assets x "
                    f"max_weight {self.max_weight:g} < 1"
                )

            asset_ids = valid.get_column(ASSET_ID).to_list()
            scores = valid.get_column("prediction").to_numpy()
            reference = _reference_vector(
                references,
                evaluation_date=evaluation_date,
                asset_ids=asset_ids,
            )
            variable = cp.Variable(valid_count)
            objective = cp.Maximize(
                scores @ variable
                - self.concentration_penalty * cp.sum_squares(variable)
                - self.turnover_penalty * cp.norm1(variable - reference)
            )
            problem = cp.Problem(
                objective,
                [cp.sum(variable) == 1, variable >= 0, variable <= self.max_weight],
            )
            attempts: list[str] = []
            solved_with: str | None = None
            for solver in ("OSQP", "CLARABEL"):
                try:
                    problem.solve(solver=solver, warm_start=True, verbose=False)
                except Exception as exc:  # solver failures must be auditable
                    attempts.append(f"{solver}: {type(exc).__name__}: {exc}")
                    continue
                attempts.append(f"{solver}: {problem.status}")
                if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                    solved_with = solver
                    break
            if solved_with is None or variable.value is None:
                raise InputValidationError(
                    f"optimizer failed at {evaluation_date}; " + "; ".join(attempts)
                )

            solution = np.asarray(variable.value, dtype=float).reshape(-1)
            violation = max(
                abs(float(solution.sum()) - 1.0),
                max(0.0, -float(solution.min())),
                max(0.0, float(solution.max()) - self.max_weight),
            )
            if not np.isfinite(solution).all() or violation > self.constraint_tolerance:
                raise InputValidationError(
                    "optimizer returned an invalid solution at "
                    f"{evaluation_date}: constraint violation {violation:.3g}"
                )
            solution = _project_capped_simplex(solution, self.max_weight)
            weight_rows.extend(
                {
                    TIME: evaluation_date,
                    ASSET_ID: asset_id,
                    "weight": float(weight),
                }
                for asset_id, weight in zip(asset_ids, solution, strict=True)
            )
            stats = problem.solver_stats
            diagnostics.append(
                {
                    TIME: evaluation_date,
                    "policy_hash": policy_hash,
                    "solver": solved_with,
                    "solver_status": str(problem.status),
                    "iterations": getattr(stats, "num_iters", None),
                    "objective": float(problem.value),
                    "constraint_violation": violation,
                }
            )

        frame = pl.DataFrame(weight_rows).sort([TIME, ASSET_ID])
        return WeightBuild(
            _weight_panel(frame, prediction),
            _empty_skipped(),
            pl.DataFrame(diagnostics).sort(TIME),
        )


def _top_n(prediction: PredictionPanel, top_n: int) -> pl.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not isinstance(prediction, PredictionPanel):
        raise TypeError("weight policies require a PredictionPanel")
    frame = validate_panel_frame(
        prediction.collect(dense=False).rename({"value": "prediction"}),
        label="predictions",
        value_columns=("prediction",),
    )
    return (
        frame.sort([TIME, "prediction"], descending=[False, True])
        .with_columns(pl.int_range(1, pl.len() + 1).over(TIME).alias("_rank"))
        .filter(pl.col("_rank") <= top_n)
        .with_columns(pl.lit(1.0).alias("_unit"))
    )


def _normalise(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return (
        frame.with_columns(
            (pl.col(column) / pl.col(column).sum().over(TIME)).alias("weight")
        )
        .select(TIME, ASSET_ID, "weight")
        .sort([TIME, ASSET_ID])
    )


def _empty_skipped() -> pl.DataFrame:
    return pl.DataFrame(schema={TIME: pl.Date, "reason": pl.String})


def _weight_panel(frame: pl.DataFrame, prediction: PredictionPanel) -> Panel:
    return Panel.from_domain(
        frame.rename({"weight": "value"}),
        prediction.domain,
        name="weights",
    )


def _weight_frame(weights: Panel) -> pl.DataFrame:
    return weights.collect(dense=False).drop_nulls("value").rename({"value": "weight"})


def _reference_weight_frame(
    reference_weights: Panel | pl.DataFrame | None,
) -> pl.DataFrame:
    if reference_weights is None:
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64}
        )
    if isinstance(reference_weights, Panel):
        frame = reference_weights.collect(dense=False).rename({"value": "weight"})
    elif isinstance(reference_weights, pl.DataFrame):
        frame = reference_weights
    else:
        raise TypeError("reference_weights must be a Panel or polars DataFrame")
    required = {TIME, ASSET_ID, "weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise InputValidationError(
            f"reference_weights is missing required columns: {missing}"
        )
    result = frame.select(TIME, ASSET_ID, "weight").with_columns(
        pl.col(TIME).cast(pl.Date, strict=False),
        pl.col(ASSET_ID).cast(pl.String),
        pl.col("weight").cast(pl.Float64, strict=False),
    )
    if result.select(pl.struct(TIME, ASSET_ID).is_duplicated().any()).item():
        raise InputValidationError(
            "reference_weights must be unique by (time, asset_id)"
        )
    invalid = result.filter(
        pl.col("weight").is_null()
        | ~pl.col("weight").is_finite()
        | (pl.col("weight") < 0)
    )
    if invalid.height:
        raise InputValidationError(
            "reference_weights.weight must be finite and nonnegative"
        )
    return result.sort([TIME, ASSET_ID])


def _reference_vector(
    reference_weights: pl.DataFrame,
    *,
    evaluation_date: object,
    asset_ids: list[str],
) -> np.ndarray:
    lookup = {
        str(row[ASSET_ID]): float(row["weight"])
        for row in reference_weights.filter(pl.col(TIME) == evaluation_date).iter_rows(
            named=True
        )
    }
    return np.asarray([lookup.get(asset_id, 0.0) for asset_id in asset_ids])


def _optimizer_policy_hash(
    policy: PredictionRegularizedOptimizerPolicy,
) -> str:
    payload = json.dumps(
        {
            "method": "prediction_regularized_optimizer",
            "concentration_penalty": policy.concentration_penalty,
            "turnover_penalty": policy.turnover_penalty,
            "max_weight": policy.max_weight,
            "constraint_tolerance": policy.constraint_tolerance,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _project_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    """Remove solver noise while preserving the simplex and upper bound."""

    if abs(values.size * cap - 1.0) <= 1e-12:
        return np.full(values.size, cap, dtype=float)
    lower = float(values.min() - cap)
    upper = float(values.max())
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        projected = np.clip(values - midpoint, 0.0, cap)
        if projected.sum() > 1.0:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(values - (lower + upper) / 2.0, 0.0, cap)
    projected /= projected.sum()
    return projected
