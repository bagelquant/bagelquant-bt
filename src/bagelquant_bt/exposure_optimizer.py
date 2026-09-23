"""Explicit long-only exposure constraints, separate from the analytic policy."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import polars as pl
from bagelquant_core import Panel, PredictionPanel

from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME, validate_panel_frame
from .portfolio import (
    PredictionRegularizedOptimizerPolicy,
    WeightBuild,
    _empty_skipped,
    _reference_vector,
    _reference_weight_frame,
    _weight_panel,
)

PREDICTION_EXPOSURE_CONSTRAINED_OPTIMIZER_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExposureBounds:
    """Absolute portfolio exposure bounds for one caller-supplied coordinate."""

    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.lower is None and self.upper is None:
            raise ValueError("exposure bounds require at least one finite boundary")
        for value in (self.lower, self.upper):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError("exposure bounds must be finite")
        if self.lower is not None and self.upper is not None:
            if self.lower > self.upper:
                raise ValueError("exposure lower bound must not exceed upper bound")


@dataclass(frozen=True, slots=True)
class PredictionExposureConstrainedOptimizerPolicy:
    """Convex quadratic/L1 optimizer with strict absolute exposure constraints.

    Callers provide point-in-time exposures and actual reference holdings at
    each decision. No market, style or industry semantics are inferred here.
    Turnover is the full sum of absolute weight changes, including liquidation
    of reference assets outside the finite Prediction cross-section. Missing
    exposures, infeasibility and solver failures are errors, never fallbacks.
    CVXPY/CLARABEL load only when ``build`` actually solves a portfolio.
    """

    concentration_penalty: float
    turnover_penalty: float
    max_weight: float
    exposure_bounds: Mapping[str, ExposureBounds] = field(default_factory=dict)
    max_turnover: float | None = None
    constraint_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        PredictionRegularizedOptimizerPolicy(
            self.concentration_penalty,
            self.turnover_penalty,
            self.max_weight,
            self.constraint_tolerance,
        )
        if self.max_turnover is not None and (
            isinstance(self.max_turnover, bool)
            or not math.isfinite(self.max_turnover)
            or self.max_turnover < 0
        ):
            raise ValueError("max_turnover must be finite and nonnegative")
        bounds = dict(self.exposure_bounds)
        for column, bound in bounds.items():
            if (
                not isinstance(column, str)
                or not column.strip()
                or column in {TIME, ASSET_ID, "prediction", "value", "weight"}
            ):
                raise ValueError(
                    "exposure column names must be nonempty and nonreserved"
                )
            if not isinstance(bound, ExposureBounds):
                raise TypeError("exposure_bounds values must be ExposureBounds")
        object.__setattr__(self, "exposure_bounds", MappingProxyType(bounds))

    def build(
        self,
        prediction: PredictionPanel,
        *,
        reference_weights: Panel | pl.DataFrame | None = None,
        exposures: pl.DataFrame | None = None,
        **_: object,
    ) -> WeightBuild:
        if not isinstance(prediction, PredictionPanel):
            raise TypeError("weight policies require a PredictionPanel")
        bounds = dict(sorted(self.exposure_bounds.items()))
        if bounds:
            if not isinstance(exposures, pl.DataFrame):
                raise InputValidationError(
                    "exposure-constrained optimizer requires exposures"
                )
            exposures = validate_panel_frame(
                exposures, label="exposures", value_columns=tuple(bounds)
            ).select(TIME, ASSET_ID, *bounds)
        references = _reference_weight_frame(reference_weights)
        if references.filter(
            pl.col(TIME).is_null() | pl.col(ASSET_ID).is_null()
        ).height:
            raise InputValidationError("reference_weights keys must be valid")
        if (
            references.group_by(TIME)
            .agg(pl.col("weight").sum())
            .filter(pl.col("weight") > 1 + self.constraint_tolerance)
            .height
        ):
            raise InputValidationError(
                "reference_weights must not exceed full investment"
            )
        predictions = prediction.collect(dense=False).rename({"value": "prediction"})
        payload = {
            "version": PREDICTION_EXPOSURE_CONSTRAINED_OPTIMIZER_VERSION,
            "concentration_penalty": self.concentration_penalty,
            "turnover_penalty": self.turnover_penalty,
            "max_weight": self.max_weight,
            "max_turnover": self.max_turnover,
            "constraint_tolerance": self.constraint_tolerance,
            "exposure_bounds": {
                key: {"lower": value.lower, "upper": value.upper}
                for key, value in bounds.items()
            },
        }
        policy_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        weight_rows = []
        diagnostics = []
        for evaluation_date in prediction.domain.times:
            valid = predictions.filter(
                (pl.col(TIME) == evaluation_date)
                & pl.col("prediction").is_not_null()
                & pl.col("prediction").is_finite()
            ).sort(ASSET_ID)
            if (
                valid.is_empty()
                or valid.height * self.max_weight < 1 - self.constraint_tolerance
            ):
                raise InputValidationError(
                    f"optimizer constraints are infeasible at {evaluation_date}: "
                    f"{valid.height} finite assets with max_weight {self.max_weight:g}"
                )
            if bounds:
                valid = valid.join(exposures, on=[TIME, ASSET_ID], how="left")
                missing = valid.filter(
                    pl.any_horizontal(
                        [
                            pl.col(key).is_null() | ~pl.col(key).is_finite()
                            for key in bounds
                        ]
                    )
                )
                if missing.height:
                    examples = ", ".join(missing[ASSET_ID].head(5).to_list())
                    raise InputValidationError(
                        f"missing or nonfinite exposures at {evaluation_date}: "
                        f"{examples}"
                    )
            assets = valid[ASSET_ID].to_list()
            scores = valid["prediction"].to_numpy()
            reference = _reference_vector(
                references, evaluation_date=evaluation_date, asset_ids=assets
            )
            forced_exit = float(
                references.filter(
                    (pl.col(TIME) == evaluation_date) & ~pl.col(ASSET_ID).is_in(assets)
                )["weight"].sum()
            )
            solution, status, iterations = self._solve(
                scores, reference, forced_exit, valid, bounds, evaluation_date
            )
            turnover = float(np.abs(solution - reference).sum() + forced_exit)
            violation = max(
                abs(float(solution.sum()) - 1.0),
                max(0.0, -float(solution.min())),
                max(0.0, float(solution.max()) - self.max_weight),
                max(0.0, turnover - self.max_turnover)
                if self.max_turnover is not None
                else 0.0,
            )
            details = {}
            for key, bound in bounds.items():
                exposure = float(valid[key].to_numpy() @ solution)
                details[f"exposure_{key}"] = exposure
                for side, limit in (("lower", bound.lower), ("upper", bound.upper)):
                    slack = (
                        None
                        if limit is None
                        else (exposure - limit if side == "lower" else limit - exposure)
                    )
                    details[f"{side}_slack_{key}"] = slack
                    if slack is not None:
                        violation = max(violation, -slack)
            if violation > self.constraint_tolerance:
                raise InputValidationError(
                    f"optimizer returned an invalid solution at {evaluation_date}: "
                    f"constraint violation {violation:.3g}"
                )
            reward = float(scores @ solution)
            concentration_cost = float(
                self.concentration_penalty * np.square(solution).sum()
            )
            turnover_cost = self.turnover_penalty * turnover
            diagnostics.append(
                {
                    TIME: evaluation_date,
                    "policy_hash": policy_hash,
                    "solver": "CLARABEL",
                    "solver_status": status,
                    "iterations": iterations,
                    "prediction_mean": float(scores.mean()),
                    "prediction_std": float(scores.std()),
                    "prediction_max_abs": float(np.abs(scores).max()),
                    "prediction_reward": reward,
                    "concentration_cost": concentration_cost,
                    "turnover_cost": turnover_cost,
                    "objective": reward - concentration_cost - turnover_cost,
                    "turnover": turnover,
                    "forced_exit_turnover": forced_exit,
                    "turnover_slack": None
                    if self.max_turnover is None
                    else self.max_turnover - turnover,
                    "constraint_violation": violation,
                    **details,
                }
            )
            weight_rows.extend(
                {TIME: evaluation_date, ASSET_ID: asset, "weight": float(weight)}
                for asset, weight in zip(assets, solution, strict=True)
            )
        return WeightBuild(
            _weight_panel(pl.DataFrame(weight_rows).sort([TIME, ASSET_ID]), prediction),
            _empty_skipped(),
            pl.DataFrame(diagnostics).sort(TIME),
        )

    def _solve(self, scores, reference, forced_exit, valid, bounds, evaluation_date):
        try:
            import cvxpy as cp
        except ImportError as error:
            raise InputValidationError(
                "exposure optimization requires bagelquant-bt[optimizer] "
                "(CVXPY/CLARABEL)"
            ) from error
        weights = cp.Variable(len(scores), nonneg=True)
        turnover = cp.norm1(weights - reference) + forced_exit
        constraints = [cp.sum(weights) == 1.0, weights <= self.max_weight]
        if self.max_turnover is not None:
            constraints.append(turnover <= self.max_turnover)
        for key, bound in bounds.items():
            exposure = valid[key].to_numpy() @ weights
            if bound.lower is not None:
                constraints.append(exposure >= bound.lower)
            if bound.upper is not None:
                constraints.append(exposure <= bound.upper)
        problem = cp.Problem(
            cp.Maximize(
                scores @ weights
                - self.concentration_penalty * cp.sum_squares(weights)
                - self.turnover_penalty * turnover
            ),
            constraints,
        )
        try:
            problem.solve(
                solver="CLARABEL",
                tol_gap_abs=min(1e-9, self.constraint_tolerance / 10),
                tol_gap_rel=min(1e-9, self.constraint_tolerance / 10),
                tol_feas=min(1e-9, self.constraint_tolerance / 10),
                max_iter=200,
            )
        except cp.error.SolverError as error:
            raise InputValidationError(
                f"exposure optimizer solver failed at {evaluation_date}: {error}"
            ) from error
        if problem.status != cp.OPTIMAL or weights.value is None:
            raise InputValidationError(
                f"exposure optimizer failed at {evaluation_date}: {problem.status}"
            )
        solution = np.asarray(weights.value, dtype=float)
        if not np.isfinite(solution).all():
            raise InputValidationError(
                f"exposure optimizer returned nonfinite weights at {evaluation_date}"
            )
        # CVXPY's nonnegative variable may contain negative numerical noise.
        # Never project/renormalize: that could violate an exposure constraint.
        return np.maximum(solution, 0.0), problem.status, problem.solver_stats.num_iters
