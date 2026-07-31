from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt.inputs import ASSET_ID, TIME, validate_prices
from bagelquant_bt.returns import _build_valuation_prices, _prepare_price_data


@pytest.mark.parametrize("sparse", [False, True])
def test_interval_price_outputs_match_dense_valuation_reference(
    sparse: bool,
) -> None:
    sessions = [date(2024, 1, day) for day in range(1, 7)]
    if sparse:
        prices = pl.DataFrame(
            {
                TIME: [
                    *sessions,
                    sessions[1],
                    sessions[2],
                    sessions[4],
                    sessions[5],
                    sessions[0],
                    sessions[2],
                    sessions[5],
                ],
                ASSET_ID: [*["a"] * 6, *["b"] * 4, "c", "d", "d"],
                "price": [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    50.0,
                    51.0,
                    55.0,
                    56.0,
                    80.0,
                    20.0,
                    25.0,
                ],
            }
        )
    else:
        prices = pl.DataFrame(
            {
                TIME: [*sessions, *sessions],
                ASSET_ID: [*["a"] * 6, *["b"] * 6],
                "price": [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    50.0,
                    51.0,
                    52.0,
                    53.0,
                    54.0,
                    55.0,
                ],
            }
        )
    observed = validate_prices(prices)
    valuation = _build_valuation_prices(observed)
    expected_returns = (
        valuation.with_columns(
            (
                pl.col("price").shift(-1).over(ASSET_ID) / pl.col("price") - 1.0
            ).alias("forward_return")
        )
        .filter(pl.col("price").is_not_null() & pl.col("forward_return").is_not_null())
        .select(TIME, ASSET_ID, "forward_return")
        .sort([TIME, ASSET_ID])
    )
    expected_gaps = (
        valuation.filter(
            pl.col("price").is_not_null() & pl.col("_observed_price").is_null()
        )
        .select(
            TIME,
            ASSET_ID,
            pl.col("_last_observed_time").alias("last_observed_time"),
            (pl.col("_session_index") - pl.col("_last_observed_session"))
            .cast(pl.Int64)
            .alias("missing_session_count"),
        )
        .sort([TIME, ASSET_ID])
    )

    actual = _prepare_price_data(observed, inputs_sorted=True)

    assert_frame_equal(actual.observed_prices, observed)
    assert_frame_equal(
        actual.forward_returns,
        expected_returns,
        check_exact=False,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert_frame_equal(actual.price_gaps, expected_gaps)
    assert_frame_equal(
        actual.valuation_prices,
        valuation.select(TIME, ASSET_ID, "price"),
    )
