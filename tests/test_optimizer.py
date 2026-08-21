from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from bagelquant_core import Domain, PredictionPanel

from bagelquant_bt import (
    PredictionRegularizedOptimizerPolicy,
    PredictionRegularizedTargetVolatilityPolicy,
)
from bagelquant_bt.exceptions import InputValidationError


def _prediction(
    values: dict[str, float | None],
    day: date = date(2024, 1, 2),
) -> PredictionPanel:
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
    assert diagnostic["policy_hash"] == (
        "274aa6cdb81f0f0f5352c8ae2e690cff463c126bbcecefee33c18ede2eeb43c9"
    )


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


def test_target_volatility_optimizer_scales_risky_sleeve_and_leaves_cash() -> None:
    evaluation_date = date(2024, 4, 1)
    prediction = PredictionPanel.from_domain(
        pl.DataFrame(
            {
                "time": [evaluation_date] * 25,
                "asset_id": [f"a{index:02d}" for index in range(25)],
                "value": [float(index) for index in range(25)],
            }
        ),
        Domain(
            calendar=[evaluation_date],
            universe=[f"a{index:02d}" for index in range(25)],
        ),
        name="prediction",
    )
    returns = pl.DataFrame(
        {
            "time": [date(2024, 3, day) for day in (1, 2, 3)],
            "gross_return": [-0.1, 0.0, 0.1],
        }
    )
    result = PredictionRegularizedTargetVolatilityPolicy(
        concentration_penalty=1.0,
        turnover_penalty=0.0,
        max_weight=0.04,
        target_annual_volatility=0.05,
        lookback_sessions=3,
        annualization=1,
    ).build(prediction, risky_sleeve_returns=returns)

    weights = result.weights.collect(dense=False).get_column("value")
    assert weights.sum() == pytest.approx(0.5)
    assert weights.max() == pytest.approx(0.02)
    diagnostic = result.diagnostics.row(0, named=True)
    assert diagnostic["realized_annual_volatility"] == pytest.approx(0.1)
    assert diagnostic["gross_exposure"] == pytest.approx(0.5)
    assert diagnostic["cash_weight"] == pytest.approx(0.5)
    assert diagnostic["volatility_observation_count"] == 3


def test_target_volatility_optimizer_normalizes_actual_stock_reference() -> None:
    evaluation_date = date(2024, 4, 1)
    reference = pl.DataFrame(
        {
            "time": [evaluation_date, evaluation_date],
            "asset_id": ["a", "b"],
            "weight": [0.42, 0.18],
        }
    )
    returns = pl.DataFrame(
        {
            "time": [date(2024, 3, day) for day in (1, 2)],
            "gross_return": [0.0, 0.0],
        }
    )
    result = PredictionRegularizedTargetVolatilityPolicy(
        concentration_penalty=0.1,
        turnover_penalty=100.0,
        max_weight=0.8,
        lookback_sessions=2,
    ).build(
        _prediction({"a": 0.0, "b": 0.0}, evaluation_date),
        reference_weights=reference,
        risky_sleeve_returns=returns,
    )

    assert result.weights.collect(dense=False).get_column(
        "value"
    ).to_list() == pytest.approx([0.7, 0.3], abs=1e-6)


def test_target_volatility_optimizer_skips_warmup_and_ignores_future_returns() -> None:
    policy = PredictionRegularizedTargetVolatilityPolicy(
        concentration_penalty=10.0,
        turnover_penalty=0.0,
        max_weight=1.0,
        target_annual_volatility=0.15,
        lookback_sessions=3,
        annualization=1,
    )
    short_history = pl.DataFrame(
        {
            "time": [date(2023, 12, 29), date(2024, 1, 1)],
            "gross_return": [0.01, -0.01],
        }
    )
    skipped = policy.build(
        _prediction({"a": 2.0, "b": -2.0}),
        risky_sleeve_returns=short_history,
    )
    assert skipped.weights.collect(dense=False).is_empty()
    assert skipped.skipped.row(0, named=True)["reason"] == (
        "insufficient_volatility_history"
    )

    causal_history = pl.DataFrame(
        {
            "time": [date(2023, 12, day) for day in (27, 28, 29)],
            "gross_return": [0.01, -0.01, 0.02],
        }
    )
    with_future_shock = pl.concat(
        [
            causal_history,
            pl.DataFrame(
                {"time": [date(2024, 1, 3)], "gross_return": [0.9]}
            ),
        ]
    )
    causal = policy.build(
        _prediction({"a": 2.0, "b": -2.0}),
        risky_sleeve_returns=causal_history,
    )
    future = policy.build(
        _prediction({"a": 2.0, "b": -2.0}),
        risky_sleeve_returns=with_future_shock,
    )
    assert causal.weights.collect(dense=False).equals(
        future.weights.collect(dense=False)
    )


def test_target_volatility_shock_changes_only_later_evaluation_date() -> None:
    first = date(2024, 4, 1)
    second = date(2024, 4, 3)
    prediction = PredictionPanel.from_domain(
        pl.DataFrame(
            {
                "time": [first, first, second, second],
                "asset_id": ["a", "b", "a", "b"],
                "value": [2.0, -2.0, 2.0, -2.0],
            }
        ),
        Domain(calendar=[first, second], universe=["a", "b"]),
        name="prediction",
    )
    result = PredictionRegularizedTargetVolatilityPolicy(
        concentration_penalty=10.0,
        turnover_penalty=0.0,
        max_weight=1.0,
        target_annual_volatility=0.15,
        lookback_sessions=2,
        annualization=1,
    ).build(
        prediction,
        risky_sleeve_returns=pl.DataFrame(
            {
                "time": [date(2024, 3, 28), date(2024, 3, 29), date(2024, 4, 2)],
                "gross_return": [-0.01, 0.01, 0.5],
            }
        ),
    )

    diagnostics = result.diagnostics.sort("time")
    assert diagnostics.row(0, named=True)["gross_exposure"] == 1.0
    assert diagnostics.row(1, named=True)["gross_exposure"] < 1.0
