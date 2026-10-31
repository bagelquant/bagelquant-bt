from __future__ import annotations

import polars as pl
import pytest

from bagelquant_bt.exceptions import InputValidationError
from bagelquant_bt.inputs import (
    validate_execution_availability,
    validate_factor,
    validate_panel_frame,
    validate_prices,
    validate_weights,
)


def test_prices_must_be_polars_dataframe() -> None:
    with pytest.raises(InputValidationError, match="prices must be"):
        validate_prices({"time": []})  # type: ignore[arg-type]


def test_prices_require_time_asset_id_price() -> None:
    with pytest.raises(InputValidationError, match="missing required columns"):
        validate_prices(pl.DataFrame({"time": ["2024-01-01"], "price": [1.0]}))


def test_prices_are_sorted_and_cloned() -> None:
    frame = pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-01"],
            "asset_id": ["a", "a"],
            "price": [2.0, 1.0],
        }
    )

    result = validate_prices(frame)

    assert result["price"].to_list() == [1.0, 2.0]


@pytest.mark.parametrize("price", [0.0, -1.0, float("inf")])
def test_prices_must_be_finite_and_positive(price: float) -> None:
    frame = pl.DataFrame(
        {"time": ["2024-01-01"], "asset_id": ["a"], "price": [price]}
    )

    with pytest.raises(
        InputValidationError,
        match=r"prices\.price must be finite and positive",
    ):
        validate_prices(frame)


def test_required_nulls_and_nans_are_removed_before_duplicate_check() -> None:
    frame = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01", None, "2024-01-02"],
            "asset_id": ["a", "a", "b", "c"],
            "price": [float("nan"), 1.0, 2.0, None],
        }
    )

    result = validate_prices(frame)

    assert result.to_dicts() == [
        {"time": result["time"][0], "asset_id": "a", "price": 1.0}
    ]


@pytest.mark.parametrize(
    ("validator", "column", "label"),
    [
        (validate_weights, "weight", "weights.weight"),
        (validate_factor, "factor", "factor.factor"),
    ],
)
def test_backtest_numeric_inputs_reject_infinite_values(
    validator,
    column: str,
    label: str,
) -> None:
    frame = pl.DataFrame(
        {"time": ["2024-01-01"], "asset_id": ["a"], column: [float("inf")]}
    )

    with pytest.raises(InputValidationError, match=rf"{label} must be finite"):
        validator(frame)


def test_execution_availability_rejects_null_rule_values() -> None:
    frame = pl.DataFrame(
        {
            "time": ["2024-01-02"],
            "asset_id": ["a"],
            "can_buy": [None],
            "can_sell": [True],
            "reason": ["suspended"],
        },
        schema_overrides={"can_buy": pl.Boolean},
    )

    with pytest.raises(
        InputValidationError,
        match="execution_availability must not contain null values",
    ):
        validate_execution_availability(frame)


def test_panel_value_columns_iterator_keeps_numeric_and_missingness_validation():
    frame = pl.DataFrame({
        "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "asset_id": ["a"] * 3,
        "value": [1.0, None, float("nan")],
    })
    result = validate_panel_frame(
        frame, label="signal", value_columns=iter(["value"])
    )
    assert result["value"].to_list() == [1.0]
    with pytest.raises(InputValidationError, match=r"signal\.value must be numeric"):
        validate_panel_frame(
            frame.with_columns(pl.lit("invalid").alias("value")),
            label="signal", value_columns=iter(["value"]),
        )
