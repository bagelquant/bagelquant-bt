from datetime import UTC, datetime

import polars as pl
import pytest

from bagelquant_bt import value_observed_account
from bagelquant_bt.exceptions import InputValidationError


def _positions(price_a: float | None, price_b: float | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asset_id": ["A", "B"],
            "quantity": [100.0, 10.0],
            "price": [price_a, price_b],
        }
    )


def test_observed_account_values_flows_without_changing_nav() -> None:
    initial = value_observed_account(
        _positions(10.0, 20.0),
        observed_at=datetime(2026, 8, 14, 15, 30, tzinfo=UTC),
        cash=800.0,
    )
    assert initial.equity == pytest.approx(2_000.0)
    assert initial.units == pytest.approx(2_000.0)
    assert initial.nav == pytest.approx(1.0)

    subscribed = value_observed_account(
        _positions(10.0, 20.0),
        observed_at=datetime(2026, 8, 17, 15, 30, tzinfo=UTC),
        cash=1_300.0,
        external_flow=500.0,
        previous=initial,
    )
    assert subscribed.equity == pytest.approx(2_500.0)
    assert subscribed.units == pytest.approx(2_500.0)
    assert subscribed.nav == pytest.approx(1.0)
    assert subscribed.flow_neutral_return == pytest.approx(0.0)


def test_observed_account_carries_stale_marks_and_recognizes_later_move() -> None:
    initial = value_observed_account(
        _positions(10.0, 20.0),
        observed_at=datetime(2026, 8, 14, 15, 30),
        cash=800.0,
    )
    stale = value_observed_account(
        _positions(None, 20.0),
        observed_at=datetime(2026, 8, 17, 15, 30),
        cash=800.0,
        previous=initial,
    )
    assert stale.equity == pytest.approx(2_000.0)
    assert stale.stale_assets == ("A",)

    resumed = value_observed_account(
        _positions(11.0, 20.0),
        observed_at=datetime(2026, 8, 18, 15, 30),
        cash=800.0,
        previous=stale,
    )
    assert resumed.equity == pytest.approx(2_100.0)
    assert resumed.nav == pytest.approx(1.05)
    assert resumed.flow_neutral_return == pytest.approx(0.05)


def test_observed_account_rejects_unpriced_new_position() -> None:
    with pytest.raises(InputValidationError, match="has no current or previous price"):
        value_observed_account(
            _positions(None, 20.0),
            observed_at=datetime(2026, 8, 14, 15, 30),
            cash=800.0,
        )
