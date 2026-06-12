from __future__ import annotations

import polars as pl
import pytest

from bagelquant_bt.exceptions import InputValidationError
from bagelquant_bt.inputs import validate_prices


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
