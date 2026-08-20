from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from bagelquant_core import (
    Domain,
    ICWeightedDecayPredictionComposer,
    ICWeightedPredictionComposer,
    IdentityPredictionComposer,
    Panel,
    PredictionComposer,
    PredictionPanel,
)

from bagelquant_bt import (
    BacktestConfig,
    EqualWeightPolicy,
    compose_prediction,
    compose_processed_prediction,
    resolve_alpha_policy,
    run_prediction_backtest,
)


def _inputs() -> tuple[pl.DataFrame, Panel, pl.DataFrame]:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(4)]
    calendar = pl.DataFrame({"time": days})
    domain = Domain(calendar=days, universe=["a", "b"])
    alpha = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [day for day in days for _ in range(2)],
                "asset_id": ["a", "b"] * len(days),
                "value": [2.0, 1.0] * len(days),
            }
        ),
        domain,
        name="alpha",
    )
    prices = pl.DataFrame(
        {
            "time": [day for day in days for _ in range(2)],
            "asset_id": ["a", "b"] * len(days),
            "price": [10.0, 10.0, 11.0, 9.0, 12.0, 8.0, 13.0, 7.0],
        }
    )
    return calendar, alpha, prices


def test_compose_and_backtest_enforce_typed_prediction_boundary() -> None:
    calendar, alpha, prices = _inputs()
    policy = resolve_alpha_policy("daily", standardization="z_score")

    prediction = compose_prediction(
        {"alpha": alpha},
        IdentityPredictionComposer(),
        calendar,
        policy,
    )
    result = run_prediction_backtest(
        prediction,
        prices,
        calendar,
        weight_policy=EqualWeightPolicy(1),
        config=BacktestConfig(initial_capital=10_000),
    )

    assert isinstance(prediction, PredictionPanel)
    assert prediction.metadata["standardization"] == "z_score"
    assert result.returns.height == 2
    with pytest.raises(TypeError, match="requires a PredictionPanel"):
        run_prediction_backtest(  # type: ignore[arg-type]
            alpha,
            prices,
            calendar,
            weight_policy=EqualWeightPolicy(1),
            config=BacktestConfig(initial_capital=10_000),
        )


def test_processed_prediction_matches_one_step_composition_without_reapplying_policy(
) -> None:
    calendar, alpha, _ = _inputs()
    policy = resolve_alpha_policy("daily", standardization="z_score")
    processed = policy.apply({"alpha": alpha}, calendar)

    direct = compose_prediction(
        {"alpha": alpha},
        IdentityPredictionComposer(),
        calendar,
        policy,
    )
    cached = compose_processed_prediction(
        processed,
        IdentityPredictionComposer(),
        calendar,
    )

    assert cached.collect(dense=False).equals(direct.collect(dense=False))
    assert cached.metadata == direct.metadata


@pytest.mark.parametrize(
    ("assets", "values"),
    [
        (["a"], [7.5]),
        (["a", "b"], [3.0, 3.0]),
        (["a", "b"], [2.0, -4.0]),
    ],
)
def test_composition_preserves_raw_prediction_values(
    assets: list[str], values: list[float]
) -> None:
    day = date(2024, 1, 2)
    calendar = pl.DataFrame({"time": [day]})
    alpha = Panel.from_domain(
        pl.DataFrame(
            {"time": [day] * len(assets), "asset_id": assets, "value": values}
        ),
        Domain(calendar=[day], universe=assets),
        name="alpha",
    )

    prediction = compose_prediction(
        {"alpha": alpha},
        IdentityPredictionComposer(),
        calendar,
        resolve_alpha_policy("daily", standardization="none"),
    )

    assert prediction.collect(dense=False).get_column("value").to_list() == values
    assert "normalization" not in prediction.metadata


def test_alpha_policy_rejects_untyped_frames() -> None:
    calendar, _, _ = _inputs()
    with pytest.raises(TypeError, match="ordinary Panel"):
        resolve_alpha_policy("daily").apply(  # type: ignore[dict-item]
            {
                "alpha": pl.DataFrame(
                    {
                        "time": [date(2024, 1, 1)],
                        "asset_id": ["a"],
                        "value": [1.0],
                    }
                )
            },
            calendar,
        )


@pytest.mark.parametrize(
    "composer",
    [
        ICWeightedPredictionComposer(1),
        ICWeightedDecayPredictionComposer(window=1, half_life=6),
    ],
    ids=("equal-period", "decayed"),
)
def test_monthly_supervised_composition_uses_completed_execution_periods(
    composer: PredictionComposer,
) -> None:
    sessions = [
        date(2024, 1, 31),
        date(2024, 2, 1),
        date(2024, 2, 29),
        date(2024, 3, 1),
        date(2024, 3, 29),
        date(2024, 4, 1),
        date(2024, 4, 30),
        date(2024, 5, 1),
    ]
    assets = ["a", "b", "c"]
    calendar = pl.DataFrame({"time": sessions})
    domain = Domain(calendar=sessions, universe=assets)
    positive = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [session for session in sessions for _ in assets],
                "asset_id": assets * len(sessions),
                "value": [1.0, 2.0, 3.0] * len(sessions),
            }
        ),
        domain,
        name="positive",
    )
    negative = Panel.from_domain(
        pl.DataFrame(
            {
                "time": [session for session in sessions for _ in assets],
                "asset_id": assets * len(sessions),
                "value": [3.0, 2.0, 1.0] * len(sessions),
            }
        ),
        domain,
        name="negative",
    )
    prices = pl.DataFrame(
        {
            "time": [session for session in sessions for _ in assets],
            "asset_id": assets * len(sessions),
            "price": [
                100.0 * (1.0 + rate) ** session_index
                for session_index, _ in enumerate(sessions)
                for rate in (0.0, 0.01, 0.02)
            ],
        }
    )
    policy = resolve_alpha_policy("month_end")

    signal = compose_prediction(
        {"positive": positive, "negative": negative},
        composer,
        calendar,
        policy,
        prices=prices,
        end=date(2024, 4, 30),
    )

    # February cannot train on its own next execution price (April 1).
    # January's label is available on March 1, so March is the first output.
    assert signal.collect(dense=False).get_column("time").unique().to_list() == [
        date(2024, 3, 29),
        date(2024, 4, 30),
    ]
