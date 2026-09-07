"""Pure, on-demand factor comparison, conditional IC and holdings exposure.

Callers own cadence, authorization, source identity and label construction.
All cross-sectional inputs describe one date, bounding temporary matrix memory.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata

from .horizon import centered_rank_book_weights, gross_one_tail_weights, hac_mean_test


def partial_rank_ic(
    target: np.ndarray,
    returns: np.ndarray,
    controls: np.ndarray | None = None,
) -> dict[str, Any]:
    """Average-tie rank partial correlation using an intercept and SVD OLS.

    Missing values use one complete-case cross-section. Collinear controls retain
    their common span; a target entirely in that span has undefined incremental IC.
    """
    x, y = np.asarray(target, dtype=float), np.asarray(returns, dtype=float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("target and returns must be equally sized vectors")
    z = np.empty((len(x), 0)) if controls is None else np.asarray(controls, dtype=float)
    if z.ndim != 2 or z.shape[0] != len(x):
        raise ValueError("controls must have one row per asset")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z).all(axis=1)
    n = int(valid.sum())
    result = {
        "ic": None,
        "sample_count": n,
        "eligible_count": len(x),
        "control_count": z.shape[1],
        "effective_rank": 0,
        "reason": None,
        "method": "partial_rank_ic" if z.shape[1] else "rank_ic",
    }
    if n < 3:
        return {**result, "reason": "insufficient_cross_section"}
    ranks = [rankdata(z[valid, j], method="average") / n for j in range(z.shape[1])]
    design = np.column_stack([np.ones(n), *ranks])
    response = np.column_stack([rankdata(x[valid]) / n, rankdata(y[valid]) / n])
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    result["effective_rank"] = int(rank)
    if n <= rank + 2:
        return {**result, "reason": "insufficient_cross_section"}
    residuals = response - design @ coefficients
    if np.std(residuals[:, 0]) <= 1e-12:
        return {**result, "reason": "no_independent_signal"}
    if np.std(residuals[:, 1]) <= 1e-12:
        return {**result, "reason": "no_residual_return_variation"}
    return {**result, "ic": float(np.clip(np.corrcoef(residuals.T)[0, 1], -1, 1))}


def incremental_ic_summary(values: list[float | None], *, horizon: int = 1) -> dict:
    """Existing Bartlett Newey-West inference with a 12-period display floor."""
    result = asdict(hac_mean_test(values, window_width=horizon))
    if result["sample_size"] < 12:
        for key in (
            "standard_error",
            "t_value",
            "p_value",
            "confidence_low",
            "confidence_high",
        ):
            result[key] = None
        result["reason"] = "insufficient_periods"
    return result


def factor_return_correlation(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    minimum_samples: int = 12,
) -> dict:
    """Pairwise-complete Pearson correlation of time/value return series."""
    for frame in (left, right):
        if frame["time"].n_unique() != frame.height:
            raise ValueError("return dates must be unique")
    paired = (
        left.select("time", pl.col("value").alias("left"))
        .join(
            right.select("time", pl.col("value").alias("right")),
            on="time",
        )
        .filter(pl.col("left").is_finite() & pl.col("right").is_finite())
        .sort("time")
    )
    result = {
        "correlation": None,
        "sample_count": paired.height,
        "start_date": paired["time"].min(),
        "end_date": paired["time"].max(),
        "reason": None,
    }
    if paired.height < minimum_samples:
        return {**result, "reason": "insufficient_samples"}
    if paired["left"].n_unique() < 2 or paired["right"].n_unique() < 2:
        return {**result, "reason": "constant_series"}
    return {
        **result,
        "correlation": float(
            np.clip(
                np.corrcoef(paired["left"].to_numpy(), paired["right"].to_numpy())[
                    0, 1
                ],
                -1,
                1,
            )
        ),
    }


def monthly_rank_returns(
    factor: pl.DataFrame, labels: pl.DataFrame, *, metric: str
) -> pl.DataFrame:
    """Gross-one Book/Tail returns over explicit execution-to-execution labels.

    Factor columns: evaluation_date, execution_date, asset_id, factor.
    Label columns: evaluation_date, asset_id, forward_return. Missing member labels
    contribute zero at their original weights, with truthful coverage counts.
    Tail reuses BT's high-signal long / low-signal short convention.
    """
    if metric not in {"book_return", "tail_return"}:
        raise ValueError("metric must be book_return or tail_return")
    weights = (
        centered_rank_book_weights(factor)
        if metric == "book_return"
        else gross_one_tail_weights(factor)
    )
    weight = "book_weight" if metric == "book_return" else "tail_weight"
    if labels.select("evaluation_date", "asset_id").is_duplicated().any():
        raise ValueError("forward-return keys must be unique")
    joined = weights.join(labels, on=["evaluation_date", "asset_id"], how="left")
    active = pl.col(weight).is_not_null() & (pl.col(weight) != 0)
    return (
        joined.group_by("evaluation_date", "execution_date")
        .agg(
            pl.col("unavailable_reason").drop_nulls().first().alias("reason"),
            active.sum().alias("expected_count"),
            (active & pl.col("forward_return").is_finite())
            .sum()
            .alias("observed_count"),
            pl.when(pl.col("forward_return").is_finite())
            .then(pl.col(weight) * pl.col("forward_return"))
            .otherwise(0.0)
            .sum()
            .alias("value"),
        )
        .with_columns(
            pl.when(pl.col("reason").is_null())
            .then(pl.col("value"))
            .otherwise(None)
            .alias("value"),
            (pl.col("observed_count") / pl.col("expected_count")).alias(
                "coverage_ratio"
            ),
        )
        .sort("evaluation_date")
    )


def holdings_factor_exposure(
    signals: pl.DataFrame,
    positions: pl.DataFrame,
    *,
    equity: float,
) -> dict:
    """One snapshot's Universe population z-score weighted by actual holdings.

    Signals: asset_id/value. Positions: asset_id/market_value. Cash has zero
    exposure. Missing holdings descriptors are excluded only from the covered
    subtotal; their weights are never redistributed or presented as zero exposure.
    """
    result = {
        "exposure": None,
        "coverage_ratio": None,
        "covered_weight": 0.0,
        "holding_weight": None,
        "complete": False,
        "reason": None,
    }
    if not np.isfinite(equity) or equity <= 0:
        return {**result, "reason": "invalid_equity"}
    for frame in (signals, positions):
        if frame["asset_id"].n_unique() != frame.height:
            raise ValueError("asset keys must be unique")
    if positions.filter(
        ~pl.col("market_value").is_finite() | pl.col("market_value").is_null()
    ).height:
        return {**result, "reason": "invalid_holdings"}
    held = positions.filter(pl.col("market_value") != 0).with_columns(
        (pl.col("market_value") / equity).alias("weight"),
    )
    total = float(held["weight"].abs().sum())
    result["holding_weight"] = total
    if total == 0:
        return {**result, "exposure": 0.0, "coverage_ratio": 1.0, "complete": True}
    finite = signals.filter(pl.col("value").is_finite())
    scale = finite["value"].std(ddof=0)
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return {**result, "reason": "constant_or_missing_signal"}
    standardized = finite.select(
        "asset_id",
        ((pl.col("value") - pl.col("value").mean()) / scale).alias("z"),
    )
    covered = held.join(standardized, on="asset_id")
    amount = float(covered["weight"].abs().sum())
    complete = covered.height == held.height
    return {
        **result,
        "exposure": float((covered["weight"] * covered["z"]).sum())
        if covered.height
        else None,
        "covered_weight": amount,
        "coverage_ratio": amount / total,
        "complete": complete,
        "reason": None if complete else "incomplete_signal_coverage",
    }
