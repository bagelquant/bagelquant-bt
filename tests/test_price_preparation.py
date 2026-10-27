"""The public price-preparation boundary cannot manufacture invalid returns."""
from datetime import date

import polars as pl
import pytest

from bagelquant_bt import InputValidationError, prepare_price_data


def test_duplicate_coordinates_cannot_masquerade_as_a_complete_price_grid():
    prices = pl.DataFrame({"time": [date(2020, 1, 1)] * 2 + [date(2020, 1, 2)] * 2,
        "asset_id": ["A", "A", "B", "B"], "price": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(InputValidationError, match="unique"):
        prepare_price_data(prices)


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf")])
def test_public_price_preparation_rejects_nonpositive_or_infinite_prices(invalid):
    prices = pl.DataFrame({"time": [date(2020, 1, 1), date(2020, 1, 2)],
        "asset_id": ["A", "A"], "price": [1.0, invalid]})
    with pytest.raises(InputValidationError, match="finite and positive"):
        prepare_price_data(prices)
