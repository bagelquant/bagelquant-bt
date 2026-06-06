"""Factor evaluation routines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .engine import _require_config, backtest_weight_frame
from .exceptions import InputValidationError
from .inputs import align_signal_and_prices
from .results import FactorEvaluationResult
from .returns import asset_close_to_close_returns


def run_factor_evaluation(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
) -> FactorEvaluationResult:
    """Evaluate a factor score DataFrame against forward returns."""

    resolved_config = _require_config(config)
    aligned_factor, aligned_prices = align_signal_and_prices(factor, prices)
    return evaluate_factor_frame(
        aligned_factor,
        aligned_prices,
        config=resolved_config,
    )


def evaluate_factor_frame(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> FactorEvaluationResult:
    """Evaluate an already materialized factor score frame."""

    aligned_factor, aligned_prices = align_signal_and_prices(factor, prices)
    forward_returns = asset_close_to_close_returns(aligned_prices).shift(-1)
    valid_dates = forward_returns.index[:-1]
    factor = aligned_factor.loc[valid_dates]
    forward_returns = forward_returns.loc[valid_dates]
    if factor.empty:
        raise InputValidationError("at least two overlapping price dates are required")

    ic = information_coefficient(
        factor,
        forward_returns,
        method=config.ic_method,
    )
    ic_std = float(ic.std(ddof=1))
    ic_mean = float(ic.mean())
    icir = ic_mean / ic_std if ic_std != 0 and not np.isnan(ic_std) else np.nan
    quantile_returns = factor_quantile_returns(
        factor,
        forward_returns,
        quantiles=config.quantiles,
    )
    quantile_cumulative = (1.0 + quantile_returns.fillna(0.0)).cumprod() - 1.0
    top_minus_bottom = quantile_returns.iloc[:, -1].sub(quantile_returns.iloc[:, 0])
    top_n_weights = top_n_equal_weights(factor, top_n=config.top_n)
    top_n_backtest = backtest_weight_frame(
        top_n_weights,
        aligned_prices,
        config=config,
    )

    return FactorEvaluationResult(
        factor=factor,
        forward_returns=forward_returns,
        ic=ic,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        quantile_returns=quantile_returns,
        quantile_cumulative_returns=quantile_cumulative,
        top_minus_bottom=top_minus_bottom,
        top_n_weights=top_n_weights,
        top_n_backtest=top_n_backtest,
    )


def information_coefficient(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    method: str = "spearman",
) -> pd.Series:
    """Compute daily cross-sectional IC."""

    values: dict[pd.Timestamp, float] = {}
    for date in factor.index:
        paired = pd.concat(
            [
                factor.loc[date].rename("factor"),
                forward_returns.loc[date].rename("forward_return"),
            ],
            axis=1,
        ).dropna()
        if len(paired) < 2:
            values[date] = np.nan
            continue
        left = paired["factor"]
        right = paired["forward_return"]
        if method == "spearman":
            left = left.rank(method="average")
            right = right.rank(method="average")
        if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
            values[date] = np.nan
            continue
        values[date] = float(left.corr(right, method="pearson"))
    return pd.Series(values, name="ic")


def factor_quantile_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    quantiles: int,
) -> pd.DataFrame:
    """Compute equal-weight daily returns by factor quantile."""

    rows: list[pd.Series] = []
    labels = [f"q{number}" for number in range(1, quantiles + 1)]
    for date in factor.index:
        scores = factor.loc[date]
        returns = forward_returns.loc[date]
        paired = pd.concat(
            [scores.rename("factor"), returns.rename("forward_return")],
            axis=1,
        ).dropna()
        row = pd.Series(np.nan, index=labels, name=date)
        if len(paired) >= quantiles:
            ranks = paired["factor"].rank(method="first")
            buckets = pd.qcut(ranks, q=quantiles, labels=labels)
            row = paired.groupby(buckets, observed=True)["forward_return"].mean()
            row = row.reindex(labels)
            row.name = date
        rows.append(row)
    return pd.DataFrame(rows, index=factor.index, columns=labels)


def top_n_equal_weights(factor: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    """Convert factor scores into long-only TOP N equal weights."""

    weights = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    for date in factor.index:
        scores = factor.loc[date].dropna().sort_values(ascending=False)
        selected = scores.head(top_n).index
        if len(selected) == 0:
            weights.loc[date] = np.nan
            continue
        weights.loc[date, selected] = 1.0 / len(selected)
    return weights
