from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from bagelquant_core import Domain, PredictionPanel

from bagelquant_bt import (
    PredictionRegularizedOptimizerPolicy,
    normalize_prediction_panel,
)
from bagelquant_bt.exceptions import InputValidationError


def _prediction(values: dict[str, float | None]) -> PredictionPanel:
    day = date(2024, 1, 2)
    assets = sorted(values)
    return PredictionPanel.from_domain(
        pl.DataFrame(
            {
                "time": [day] * len(assets),
                "asset_id": assets,
                "value": [values[asset] for asset in assets],
            }
        ),
        Domain(calendar=[day], universe=assets),
        name="prediction",
    )


def test_prediction_normalization_uses_population_zscore_and_metadata() -> None:
    normalized = normalize_prediction_panel(_prediction({"a": 1.0, "b": 2.0, "c": 3.0}))

    assert normalized.collect(dense=False).get_column(
        "value"
    ).to_list() == pytest.approx([-1.2247448714, 0.0, 1.2247448714])
    assert normalized.metadata["normalization"] == {
        "method": "cross_sectional_zscore",
        "finite_values_only": True,
        "ddof": 0,
        "minimum_valid_count": 2,
        "zero_variance_action": "fail_date",
    }


@pytest.mark.parametrize(
    "values",
    [
        {"a": 1.0, "b": None},
        {"a": 1.0, "b": 1.0},
    ],
)
def test_prediction_normalization_strictly_fails_bad_cross_sections(
    values: dict[str, float | None],
) -> None:
    with pytest.raises(InputValidationError, match="failed dates: 2024-01-02"):
        normalize_prediction_panel(_prediction(values))


def test_prediction_normalization_reports_snapshot_with_no_finite_rows() -> None:
    first = date(2024, 1, 2)
    missing = date(2024, 1, 3)
    prediction = PredictionPanel.from_domain(
        pl.DataFrame(
            {
                "time": [first, first, missing, missing],
                "asset_id": ["a", "b", "a", "b"],
                "value": [1.0, 2.0, None, None],
            }
        ),
        Domain(calendar=[first, missing], universe=["a", "b"]),
        name="prediction",
    )

    with pytest.raises(InputValidationError, match="failed dates: 2024-01-03"):
        normalize_prediction_panel(prediction)


def test_optimizer_matches_hand_calculated_two_asset_solution() -> None:
    result = PredictionRegularizedOptimizerPolicy(
        concentration_penalty=10.0,
        turnover_penalty=0.0,
        max_weight=1.0,
    ).build(_prediction({"a": 2.0, "b": -2.0}))

    assert result.weights.collect(dense=False).get_column(
        "value"
    ).to_list() == pytest.approx([0.6, 0.4], abs=1e-6)
    diagnostic = result.diagnostics.row(0, named=True)
    assert diagnostic["solver"] in {"OSQP", "CLARABEL"}
    assert diagnostic["constraint_violation"] <= 1e-7
    assert len(diagnostic["policy_hash"]) == 64


def test_optimizer_honors_actual_reference_and_forces_missing_asset_exit() -> None:
    day = date(2024, 1, 2)
    reference = pl.DataFrame(
        {
            "time": [day, day, day],
            "asset_id": ["a", "b", "c"],
            "weight": [0.7, 0.2, 0.1],
        }
    )
    result = PredictionRegularizedOptimizerPolicy(
        concentration_penalty=0.1,
        turnover_penalty=100.0,
        max_weight=0.8,
    ).build(
        _prediction({"a": 0.0, "b": 0.0, "c": None}),
        reference_weights=reference,
    )

    weights = result.weights.collect(dense=False)
    assert weights.get_column("asset_id").to_list() == ["a", "b"]
    assert weights.get_column("value").to_list() == pytest.approx([0.7, 0.3], abs=1e-6)


def test_optimizer_accepts_exact_twenty_five_asset_four_percent_boundary() -> None:
    assets = {f"a{index:02d}": float(index) for index in range(25)}
    result = PredictionRegularizedOptimizerPolicy(
        concentration_penalty=1.0,
        turnover_penalty=0.0,
        max_weight=0.04,
    ).build(_prediction(assets))

    assert result.weights.collect(dense=False).get_column(
        "value"
    ).to_list() == pytest.approx(
        [0.04] * 25,
        abs=1e-7,
    )


def test_optimizer_fails_infeasible_cap_without_fallback_policy() -> None:
    with pytest.raises(InputValidationError, match="constraints are infeasible"):
        PredictionRegularizedOptimizerPolicy(
            concentration_penalty=1.0,
            turnover_penalty=0.0,
            max_weight=0.04,
        ).build(_prediction({f"a{index:02d}": float(index) for index in range(24)}))


def test_optimizer_projects_finite_solver_noise_before_strict_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cvxpy as cp

    original_solve = cp.Problem.solve

    def noisy_solve(problem, *args, **kwargs):
        result = original_solve(problem, *args, **kwargs)
        variable = problem.variables()[0]
        variable._value = variable.value + 8e-7
        return result

    monkeypatch.setattr(cp.Problem, "solve", noisy_solve)
    result = PredictionRegularizedOptimizerPolicy(
        concentration_penalty=10.0,
        turnover_penalty=0.0,
        max_weight=1.0,
    ).build(_prediction({"a": 2.0, "b": -2.0}))

    weights = result.weights.collect(dense=False).get_column("value")
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    diagnostic = result.diagnostics.row(0, named=True)
    assert diagnostic["raw_solver_constraint_violation"] > 1e-7
    assert diagnostic["constraint_violation"] <= 1e-7
