from datetime import date

import numpy as np
import polars as pl
import pytest
from bagelquant_core import Domain, PredictionPanel

from bagelquant_bt import (
    ExposureBounds,
    PredictionExposureConstrainedOptimizerPolicy,
    PredictionRegularizedOptimizerPolicy,
)
from bagelquant_bt.exceptions import InputValidationError

DAY = date(2024, 1, 2)


def prediction(values=(2.0, -2.0)):
    assets = [chr(97 + i) for i in range(len(values))]
    return PredictionPanel.from_domain(
        pl.DataFrame(
            {"time": [DAY] * len(values), "asset_id": assets, "value": values},
            schema_overrides={"value": pl.Float64},
        ),
        Domain(calendar=[DAY], universe=assets),
    )


def exposures(**columns):
    n = len(next(iter(columns.values())))
    return pl.DataFrame(
        {"time": [DAY] * n, "asset_id": [chr(97 + i) for i in range(n)], **columns}
    )


def policy(**kwargs):
    return PredictionExposureConstrainedOptimizerPolicy(
        concentration_penalty=kwargs.pop("concentration_penalty", 10),
        turnover_penalty=kwargs.pop("turnover_penalty", 0),
        max_weight=kwargs.pop("max_weight", 1),
        **kwargs,
    )


def test_hand_calculated_unconstrained_solution_and_objective():
    result = policy().build(prediction())
    assert result.weights.collect()["value"].to_list() == pytest.approx(
        [0.6, 0.4], abs=1e-7
    )
    row = result.diagnostics.row(0, named=True)
    assert row["prediction_reward"] == pytest.approx(0.4)
    assert row["concentration_cost"] == pytest.approx(5.2)
    assert row["objective"] == pytest.approx(-4.8)
    assert row["solver_status"] == "optimal"
    assert row["prediction_std"] == 2.0


def test_beta_style_and_industry_constraints_are_generic_absolute_coordinates():
    result = policy(
        exposure_bounds={
            "market_beta": ExposureBounds(lower=1.0, upper=1.0),
            "style_size": ExposureBounds(lower=0.0, upper=0.2),
            "industry_1": ExposureBounds(upper=0.55),
        }
    ).build(
        prediction(),
        exposures=exposures(
            market_beta=[2.0, 0.0], style_size=[1.0, -1.0], industry_1=[1.0, 0.0]
        ),
    )
    assert result.weights.collect()["value"].to_list() == pytest.approx(
        [0.5, 0.5], abs=1e-7
    )
    row = result.diagnostics.row(0, named=True)
    assert row["exposure_market_beta"] == pytest.approx(1.0)
    assert row["upper_slack_industry_1"] == pytest.approx(0.05)
    assert row["constraint_violation"] <= 1e-7


def test_turnover_cap_includes_forced_exits_from_prediction_universe():
    reference = pl.DataFrame(
        {"time": [DAY] * 3, "asset_id": ["a", "b", "c"], "weight": [0.5, 0.3, 0.2]}
    )
    result = policy(max_turnover=0.4, turnover_penalty=0.5).build(
        prediction(), reference_weights=reference
    )
    row = result.diagnostics.row(0, named=True)
    assert row["forced_exit_turnover"] == pytest.approx(0.2)
    assert row["turnover"] == pytest.approx(0.4, abs=1e-7)
    assert row["turnover_cost"] == pytest.approx(0.2, abs=1e-7)
    with pytest.raises(InputValidationError, match=r"2024-01-02.*infeasible"):
        policy(max_turnover=0.39).build(prediction(), reference_weights=reference)


def test_initial_investment_requires_turnover_budget_one():
    with pytest.raises(InputValidationError, match="infeasible"):
        policy(max_turnover=0.99).build(prediction())
    assert policy(max_turnover=1).build(prediction()).diagnostics["turnover"][
        0
    ] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_missing_exposures_never_become_zero(bad):
    with pytest.raises(InputValidationError, match=r"exposures.*2024-01-02"):
        policy(exposure_bounds={"beta": ExposureBounds(upper=1.0)}).build(
            prediction(), exposures=exposures(beta=[0.5, bad])
        )


def test_infeasible_exposure_fails_without_relaxation():
    with pytest.raises(InputValidationError, match=r"2024-01-02.*infeasible"):
        policy(exposure_bounds={"beta": ExposureBounds(upper=0.5)}).build(
            prediction(), exposures=exposures(beta=[1.0, 2.0])
        )


def test_cap_single_asset_and_no_finite_predictions():
    assert policy().build(prediction([1.0])).weights.collect()[
        "value"
    ].to_list() == pytest.approx([1.0])
    with pytest.raises(InputValidationError, match="infeasible"):
        policy(max_weight=0.4).build(prediction())
    with pytest.raises(InputValidationError, match="infeasible"):
        policy().build(prediction([None, None]))


def test_exposure_bounds_are_immutable_and_validate_keys():
    bounds = {"beta": ExposureBounds(upper=1.0)}
    instance = policy(exposure_bounds=bounds)
    bounds.clear()
    assert "beta" in instance.exposure_bounds
    with pytest.raises(TypeError):
        instance.exposure_bounds["beta"] = ExposureBounds(upper=2.0)
    with pytest.raises(ValueError):
        ExposureBounds(lower=2.0, upper=1.0)
    with pytest.raises(ValueError):
        ExposureBounds()
    with pytest.raises(ValueError):
        policy(max_turnover=-1.0)
    with pytest.raises(ValueError):
        policy(exposure_bounds={"value": ExposureBounds(upper=1.0)})


def test_matches_unchanged_analytic_policy_when_risk_bounds_disabled():
    rng = np.random.default_rng(78)
    source = prediction(rng.normal(size=12))
    old = PredictionRegularizedOptimizerPolicy(1.0, 0.1, 0.2).build(source)
    new = policy(concentration_penalty=1.0, turnover_penalty=0.1, max_weight=0.2).build(
        source
    )
    assert new.weights.collect()["value"].to_list() == pytest.approx(
        old.weights.collect()["value"].to_list(), abs=2e-6
    )


def test_solver_failure_is_explicit(monkeypatch):
    import cvxpy as cp

    def fail(*args, **kwargs):
        raise cp.error.SolverError("test solver failure")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    with pytest.raises(InputValidationError, match="solver failed at 2024-01-02"):
        policy().build(prediction())
