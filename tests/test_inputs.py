from __future__ import annotations

import pandas as pd
import pytest

from bagelquant_bt.exceptions import InputValidationError
from bagelquant_bt.inputs import align_signal_and_prices, validate_signal_frame


class DataBacked:
    def __init__(self, data):
        self.data = data


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [1.0, 2.0], "b": [3.0, 4.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )


def test_signal_must_be_dataframe() -> None:
    with pytest.raises(InputValidationError, match="signal must be"):
        validate_signal_frame(DataBacked(frame()))  # type: ignore[arg-type]


def test_signal_dataframe_is_defensive_copy() -> None:
    source = frame()
    result = validate_signal_frame(source)
    result.loc[pd.Timestamp("2024-01-02"), "a"] = 99.0

    assert source.loc[pd.Timestamp("2024-01-02"), "a"] == 1.0


def test_rejects_nonnumeric_signal() -> None:
    bad = pd.DataFrame({"a": ["x"]}, index=pd.to_datetime(["2024-01-02"]))

    with pytest.raises(InputValidationError, match="fully numeric"):
        validate_signal_frame(bad)


def test_prices_must_be_dataframe() -> None:
    with pytest.raises(InputValidationError, match="prices must be"):
        align_signal_and_prices(frame(), DataBacked(frame()))  # type: ignore[arg-type]


def test_alignment_intersects_dates_and_assets() -> None:
    signal = pd.DataFrame(
        {"b": [1.0, 2.0], "c": [3.0, 4.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    prices = pd.DataFrame(
        {"a": [10.0, 11.0], "b": [20.0, 22.0]},
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )

    aligned_signal, aligned_prices = align_signal_and_prices(signal, prices)

    assert aligned_signal.index.tolist() == [pd.Timestamp("2024-01-03")]
    assert aligned_signal.columns.tolist() == ["b"]
    assert aligned_prices.loc[pd.Timestamp("2024-01-03"), "b"] == 20.0
