"""Input validation helpers."""

from __future__ import annotations

import pandas as pd

from .exceptions import InputValidationError


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate daily close prices and return a normalized defensive copy."""

    if not isinstance(prices, pd.DataFrame):
        raise InputValidationError("prices must be a pandas DataFrame")
    frame = prices.copy(deep=True)
    _validate_numeric_frame(frame, label="prices")
    if frame.isna().all(axis=None):
        raise InputValidationError("prices must contain at least one non-missing value")
    return _normalize_index(frame, label="prices")


def align_signal_and_prices(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align signal/factor values and prices to common dates and assets."""

    signal = validate_signal_frame(signal)
    prices = validate_prices(prices)
    common_dates = signal.index.intersection(prices.index).sort_values()
    common_assets = prices.columns.intersection(signal.columns)

    if common_dates.empty:
        raise InputValidationError("signal and prices have no overlapping dates")
    if common_assets.empty:
        raise InputValidationError("signal and prices have no overlapping assets")

    return (
        signal.reindex(index=common_dates, columns=common_assets),
        prices.reindex(index=common_dates, columns=common_assets),
    )


def validate_signal_frame(signal: pd.DataFrame) -> pd.DataFrame:
    """Validate weights or factor scores and return a normalized defensive copy."""

    if not isinstance(signal, pd.DataFrame):
        raise InputValidationError("signal must be a pandas DataFrame")
    frame = signal.copy(deep=True)
    _validate_numeric_frame(frame, label="signal")
    return _normalize_index(frame, label="signal")


def _validate_numeric_frame(frame: pd.DataFrame, *, label: str) -> None:
    if frame.index.nlevels != 1 or frame.columns.nlevels != 1:
        raise InputValidationError(f"{label} must have 1D index and columns")
    if frame.index.has_duplicates:
        raise InputValidationError(f"{label} dates must be unique")
    if frame.columns.has_duplicates:
        raise InputValidationError(f"{label} assets must be unique")
    numeric_columns = frame.select_dtypes(include="number").columns
    if len(numeric_columns) != len(frame.columns):
        raise InputValidationError(f"{label} must be fully numeric")


def _normalize_index(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    normalized.index = pd.DatetimeIndex(pd.to_datetime(normalized.index))
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize().as_unit("ns")
    if normalized.index.has_duplicates:
        raise InputValidationError(
            f"{label} dates must remain unique after normalization"
        )
    if not normalized.index.is_monotonic_increasing:
        normalized = normalized.sort_index()
    return normalized
