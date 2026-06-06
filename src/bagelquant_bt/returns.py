"""Return-series utilities."""

from __future__ import annotations

import pandas as pd


def asset_close_to_close_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily close-to-close asset returns."""

    return prices.pct_change(fill_method=None)


def align_weights_to_forward_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align weights at t with returns from t to t+1."""

    returns = asset_close_to_close_returns(prices).shift(-1)
    valid_dates = returns.index[:-1]
    return weights.loc[valid_dates], returns.loc[valid_dates]


def portfolio_returns(
    weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """Compute row-wise weighted portfolio returns."""

    aligned_returns = forward_returns.where(weights.notna())
    return weights.fillna(0.0).mul(aligned_returns.fillna(0.0)).sum(axis=1)


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compound daily returns into cumulative return path."""

    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def value_path(returns: pd.Series, *, initial_capital: float) -> pd.Series:
    """Compound daily returns into a portfolio value path."""

    return initial_capital * (1.0 + returns.fillna(0.0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    """Compute drawdown from a daily return series."""

    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0
