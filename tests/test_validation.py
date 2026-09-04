from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bagelquant_bt import (
    score_horizon_ic_validation,
    score_ic_validation,
    score_top_n_performance,
    select_top_n_stable,
    top_n_monthly_performance,
)


def test_horizon_ic_validation_is_frequency_explicit_and_overlap_robust() -> None:
    rows: list[tuple[date, str, float, float]] = []
    for day in range(1, 9):
        current = date(2024, 1, day)
        rows.extend(
            [(current, "A", 1.0, float(day)), (current, "B", 2.0, float(day + 1))]
        )

    result = score_horizon_ic_validation(
        _prediction_frame(rows),
        horizon_sessions=5,
        annualization=240,
        minimum_valid_periods=6,
        objective="icir",
    )

    assert result.horizon_sessions == 5
    assert result.annualization == 240
    assert result.score.valid
    assert result.score.score == 0.0
    assert result.hac_lag >= 4
    assert result.cohort_count == 5


@pytest.mark.parametrize(
    ("horizon", "annualization"),
    [(0, 240), (1, 0), (True, 240)],
)
def test_horizon_ic_validation_rejects_implicit_or_invalid_frequency(
    horizon: object,
    annualization: int,
) -> None:
    with pytest.raises(ValueError):
        score_horizon_ic_validation(
            _prediction_frame([]),
            horizon_sessions=horizon,  # type: ignore[arg-type]
            annualization=annualization,
            minimum_valid_periods=1,
        )


def _prediction_frame(rows: list[tuple[date, str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "time": pl.Date,
            "asset_id": pl.String,
            "prediction": pl.Float64,
            "target": pl.Float64,
        },
        orient="row",
    )


def test_undefined_ic_months_are_excluded_instead_of_scored_as_zero() -> None:
    jan = date(2024, 1, 31)
    feb = date(2024, 2, 29)
    result = score_ic_validation(
        _prediction_frame(
            [
                (jan, "A", 1.0, 1.0),
                (jan, "B", 1.0, 2.0),
                (feb, "A", 1.0, 1.0),
                (feb, "B", 2.0, 2.0),
            ]
        ),
        minimum_valid_months=1,
    )

    assert result.valid
    assert result.score == pytest.approx(1.0)
    assert result.valid_months == 1
    assert result.monthly_ic[0].reason == "constant_prediction"


def test_candidate_fails_when_valid_month_count_is_below_threshold() -> None:
    day = date(2024, 1, 31)
    result = score_ic_validation(
        _prediction_frame([(day, "A", 1.0, 1.0), (day, "B", 2.0, 2.0)]),
        minimum_valid_months=2,
    )

    assert not result.valid
    assert result.score is None
    assert result.reason == "insufficient_valid_validation_months"


def test_expected_month_without_pairs_is_explicitly_invalid() -> None:
    first = date(2024, 1, 31)
    missing = date(2024, 2, 29)
    result = score_ic_validation(
        _prediction_frame(
            [(first, "A", 1.0, 1.0), (first, "B", 2.0, 2.0)]
        ),
        minimum_valid_months=1,
        expected_periods=(first, missing),
    )

    assert result.valid
    assert result.monthly_ic[1].time == missing
    assert result.monthly_ic[1].reason == "insufficient_pairs"
    assert result.monthly_ic[1].pair_count == 0


def test_icir_is_zero_when_valid_monthly_ic_has_zero_deviation() -> None:
    rows = []
    for month in (1, 2):
        current = date(2024, month, 28)
        rows.extend([(current, "A", 1.0, 1.0), (current, "B", 2.0, 2.0)])

    result = score_ic_validation(
        _prediction_frame(rows), minimum_valid_months=2, objective="icir"
    )

    assert result.valid
    assert result.score == 0.0


def test_top_n_ties_use_asset_id_and_record_cutoff_audit() -> None:
    day = date(2024, 1, 31)
    frame = _prediction_frame(
        [
            (day, "C", 2.0, 0.0),
            (day, "B", 2.0, 0.0),
            (day, "A", 3.0, 0.0),
            (day, "D", 1.0, 0.0),
        ]
    )

    result = select_top_n_stable(frame, top_n=2)

    assert result.selections.get_column("asset_id").to_list() == ["A", "B"]
    assert result.audit.row(0, named=True) == {
        "time": day,
        "valid": True,
        "reason": None,
        "eligible_count": 4,
        "cutoff_score": 2.0,
        "cutoff_tie_count": 2,
        "tie_excluded_count": 1,
    }


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ([('A', 1.0)], "underfilled_top_n"),
        ([('A', 1.0), ('B', 1.0)], "constant_prediction"),
    ],
)
def test_top_n_rejects_underfilled_and_constant_months(rows, reason: str) -> None:
    day = date(2024, 1, 31)
    frame = _prediction_frame(
        [(day, asset_id, prediction, 0.0) for asset_id, prediction in rows]
    )

    result = select_top_n_stable(frame, top_n=2)

    assert result.selections.is_empty()
    assert result.audit.get_column("reason").to_list() == [reason]


def test_turnover_regularization_is_in_addition_to_net_cost_returns() -> None:
    performance = pl.DataFrame(
        {
            "time": [date(2024, month, 28) for month in range(1, 4)],
            "net_return": [0.01, 0.02, 0.03],
            "turnover": [0.2, 0.4, 0.6],
        }
    )
    sharpe = score_top_n_performance(
        performance,
        minimum_valid_months=3,
        objective="top_n_net_sharpe",
    )
    regularized = score_top_n_performance(
        performance,
        minimum_valid_months=3,
        objective="top_n_net_sharpe_with_turnover_regularization",
        turnover_penalty=0.5,
    )

    assert regularized.score == pytest.approx(sharpe.score - 0.5 * 0.4)


def test_top_n_monthly_performance_uses_canonical_absolute_turnover() -> None:
    jan = date(2024, 1, 31)
    feb = date(2024, 2, 29)
    predictions = pl.DataFrame(
        {
            "time": [jan, jan, feb, feb],
            "asset_id": ["A", "B", "A", "B"],
            "prediction": [2.0, 1.0, 1.0, 2.0],
            "forward_return": [0.1, 0.0, 0.0, 0.2],
        }
    )

    performance, _ = top_n_monthly_performance(
        predictions, top_n=1, cost_rate_per_turnover=0.01
    )

    assert performance.get_column("turnover").to_list() == [1.0, 2.0]
    assert performance.get_column("net_return").to_list() == pytest.approx(
        [0.09, 0.18]
    )


def test_top_n_monthly_performance_applies_per_asset_minimum_commission() -> None:
    day = date(2024, 1, 31)
    predictions = pl.DataFrame(
        {
            "time": [day, day],
            "asset_id": ["A", "B"],
            "prediction": [2.0, 1.0],
            "forward_return": [0.1, 0.0],
        }
    )

    performance, _ = top_n_monthly_performance(
        predictions,
        top_n=1,
        commission_rate=0.01,
        minimum_fee=5.0,
        initial_capital=100.0,
    )

    assert performance.get_column("net_return").to_list() == pytest.approx([0.05])
