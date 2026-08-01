from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from bagelquant_core import (
    Domain,
    ICWeightedSignalComposer,
    IdentitySignalComposer,
    Panel,
    SignalPanel,
)

from bagelquant_bt import (
    BacktestConfig,
    EqualWeightPolicy,
    compose_signal,
    resolve_signal_date_policy,
    run_signal_backtest,
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


def test_compose_and_backtest_enforce_typed_signal_boundary() -> None:
    calendar, alpha, prices = _inputs()
    policy = resolve_signal_date_policy("daily")

    signal = compose_signal(
        {"alpha": alpha},
        IdentitySignalComposer(),
        calendar,
        policy,
    )
    result = run_signal_backtest(
        signal,
        prices,
        calendar,
        policy,
        portfolio_policy=EqualWeightPolicy(1),
        config=BacktestConfig(initial_capital=10_000),
    )

    assert isinstance(signal, SignalPanel)
    assert signal.metadata["standardization"] == "zscore"
    assert result.returns.height == 2
    with pytest.raises(TypeError, match="requires a SignalPanel"):
        run_signal_backtest(  # type: ignore[arg-type]
            alpha,
            prices,
            calendar,
            policy,
            config=BacktestConfig(initial_capital=10_000),
        )


def test_signal_date_policy_rejects_untyped_frames() -> None:
    calendar, _, _ = _inputs()
    with pytest.raises(TypeError, match="requires a SignalPanel"):
        resolve_signal_date_policy("daily").select(  # type: ignore[arg-type]
            pl.DataFrame(
                {"time": [date(2024, 1, 1)], "asset_id": ["a"], "value": [1.0]}
            ),
            calendar,
        )


def test_monthly_supervised_composition_uses_completed_execution_periods() -> None:
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
    policy = resolve_signal_date_policy("month_end")

    signal = compose_signal(
        {"positive": positive, "negative": negative},
        ICWeightedSignalComposer(1),
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
