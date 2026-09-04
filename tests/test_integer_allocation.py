from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

import bagelquant_bt.allocation as allocation_module
from bagelquant_bt import InputValidationError, allocate_integer_positions


def test_integer_allocation_maximizes_budget_then_tracks_targets() -> None:
    weights = pl.DataFrame(
        {
            "asset_id": ["A", "B", "C"],
            "weight": [0.4, 0.35, 0.25],
        }
    )
    prices = pl.DataFrame({"asset_id": ["A", "B", "C"], "price": [11.0, 7.0, 3.0]})
    lots = pl.DataFrame({"asset_id": ["A", "B", "C"], "lot_size": [100, 100, 100]})

    result = allocate_integer_positions(
        weights,
        prices,
        total_notional=10_000.0,
        lot_sizes=lots,
    )

    assert result.allocated_notional == 10_000.0
    assert result.residual_cash == 0.0
    assert result.positions.select("asset_id", "target_quantity").to_dicts() == [
        {"asset_id": "A", "target_quantity": 400},
        {"asset_id": "B", "target_quantity": 500},
        {"asset_id": "C", "target_quantity": 700},
    ]


def test_integer_allocation_allows_at_most_one_lot_above_each_target() -> None:
    result = allocate_integer_positions(
        pl.DataFrame({"asset_id": ["A", "B"], "weight": [0.5, 0.5]}),
        pl.DataFrame({"asset_id": ["A", "B"], "price": [60.0, 40.0]}),
        total_notional=10_000.0,
        lot_sizes=pl.DataFrame({"asset_id": ["A", "B"], "lot_size": [100, 100]}),
    )
    rows = {row["asset_id"]: row for row in result.positions.to_dicts()}

    assert result.allocated_notional <= result.stock_budget
    assert rows["A"]["target_quantity"] <= 100
    assert rows["B"]["target_quantity"] <= 200


def test_integer_allocation_respects_exposure_and_minimum_positions() -> None:
    result = allocate_integer_positions(
        pl.DataFrame({"asset_id": ["A", "B"], "weight": [0.3, 0.2]}),
        pl.DataFrame({"asset_id": ["A", "B"], "price": [10.0, 10.0]}),
        total_notional=20_000.0,
        lot_sizes=pl.DataFrame({"asset_id": ["A", "B"], "lot_size": [100, 100]}),
        minimum_quantities=pl.DataFrame({"asset_id": ["A"], "minimum_quantity": [300]}),
    )

    assert result.stock_exposure == pytest.approx(0.5)
    assert result.stock_budget == 10_000.0
    assert result.allocated_notional == 10_000.0
    assert (
        result.positions.filter(pl.col("asset_id") == "A").item(0, "target_quantity")
        >= 300
    )


def test_integer_allocation_is_row_order_independent() -> None:
    weights = pl.DataFrame({"asset_id": ["C", "A", "B"], "weight": [0.2, 0.4, 0.4]})
    prices = pl.DataFrame({"asset_id": ["B", "C", "A"], "price": [13.0, 7.0, 11.0]})

    first = allocate_integer_positions(
        weights,
        prices,
        total_notional=10_000.0,
        lot_sizes=pl.DataFrame(
            {"asset_id": ["A", "B", "C"], "lot_size": [100, 100, 100]}
        ),
    )
    second = allocate_integer_positions(
        weights.reverse(),
        prices.reverse(),
        total_notional=10_000.0,
        lot_sizes=pl.DataFrame(
            {"asset_id": ["C", "B", "A"], "lot_size": [100, 100, 100]}
        ),
    )

    assert first.positions.equals(second.positions)
    assert first.residual_cash == second.residual_cash


def test_integer_allocation_rejects_unfunded_minimum_positions() -> None:
    with pytest.raises(InputValidationError, match="minimum positions exceed"):
        allocate_integer_positions(
            pl.DataFrame({"asset_id": ["A"], "weight": [0.5]}),
            pl.DataFrame({"asset_id": ["A"], "price": [10.0]}),
            total_notional=1_000.0,
            minimum_quantities=pl.DataFrame(
                {"asset_id": ["A"], "minimum_quantity": [100]}
            ),
        )


def test_integer_allocation_preserves_solver_status_and_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocation_module,
        "milp",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False,
            x=None,
            status=4,
            message="HiGHS model error",
        ),
    )

    with pytest.raises(
        InputValidationError,
        match=r"maximize the stock budget.*status=4.*HiGHS model error",
    ):
        allocate_integer_positions(
            pl.DataFrame({"asset_id": ["A"], "weight": [1.0]}),
            pl.DataFrame({"asset_id": ["A"], "price": [10.0]}),
            total_notional=10_000.0,
        )
