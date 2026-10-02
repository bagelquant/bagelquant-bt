"""Saved-result library inference and implementation diagnostics.

Callers freeze the trial family and source receipts. These functions neither fit
models nor simulate accounts. Deflated Sharpe follows Bailey/Lopez de Prado
(2014), https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict

import numpy as np
import polars as pl

from .horizon import benjamini_hochberg, hac_mean_test


def library_endpoint_tests(
    trials: Sequence[Mapping], *, q_max: float = 0.10
) -> list[dict]:
    """Apply two-sided BH across every valid preregistered primary endpoint.

    Abandonment never removes an available statistic from the test family.
    Missing statistics keep null p/q values and an explicit reason.
    """
    if not 0 < q_max < 1:
        raise ValueError("q_max must be between zero and one")
    rows = []
    for trial in trials:
        if trial.get("direction", "positive") not in {"positive", "negative"}:
            raise ValueError("direction must be positive or negative")
        minimum = float(trial.get("minimum_effect", 0))
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError("minimum_effect must be finite and nonnegative")
        result = dict(trial)
        statistic = dict(trial.get("statistic", {}))
        p = statistic.get("p_value")
        valid = (
            p is not None
            and math.isfinite(float(p))
            and 0 <= p <= 1
            and statistic.get("mean") is not None
            and math.isfinite(float(statistic["mean"]))
            and not statistic.get("unavailable_reason", statistic.get("reason"))
        )
        result.update(
            p_value=float(p) if valid else None,
            mean=statistic.get("mean"),
            unavailable_reason=None
            if valid
            else statistic.get("unavailable_reason")
            or statistic.get("reason")
            or trial.get("unavailable_reason")
            or "primary_endpoint_unavailable",
        )
        rows.append(result)
    for row, q in zip(
        rows, benjamini_hochberg([row["p_value"] for row in rows]), strict=True
    ):
        mean = row["mean"]
        direction = row.get("direction", "positive")
        signed = (
            None
            if mean is None
            else float(mean) * (1 if direction == "positive" else -1)
        )
        row.update(
            q_value=q,
            q_max=q_max,
            family_size=sum(item["p_value"] is not None for item in rows),
            passes=q is not None
            and q <= q_max
            and signed is not None
            and signed > 0
            and signed >= float(row.get("minimum_effect", 0)),
        )
    return rows


def deflated_sharpe(
    returns: pl.DataFrame,
    *,
    target: str,
    trials: Sequence[str],
    minimum_sessions: int = 240,
) -> dict:
    """Compute auxiliary DSR on one common finite sample of explicitly named trials.

    Returns are long-form (time, trial, return). Effective trials use the stated
    mean-correlation heuristic N=rho+(1-rho)*M. It is an estimate, not a count
    of all undocumented historical experiments.
    """
    scope = list(dict.fromkeys(trials))
    result = {
        "status": "unavailable",
        "trial_scope": scope,
        "sample_count": 0,
        "effective_trials_method": "mean_correlation",
        "source": "Bailey_Lopez_de_Prado_2014",
    }
    if target not in scope or len(scope) < 3:
        return {**result, "reason": "at_least_three_declared_trials_required"}
    if returns.select(pl.struct("time", "trial").is_duplicated().any()).item():
        raise ValueError("trial return keys must be unique")
    wide = (
        returns.filter(pl.col("trial").is_in(scope))
        .pivot(on="trial", index="time", values="return")
        .sort("time")
    )
    if set(scope) - set(wide.columns):
        return {**result, "reason": "trial_returns_missing"}
    sample = (
        wide.filter(
            pl.all_horizontal(pl.col(scope).is_not_null() & pl.col(scope).is_finite())
        )
        .select(scope)
        .to_numpy()
    )
    result["sample_count"] = len(sample)
    if len(sample) < minimum_sessions:
        return {**result, "reason": "insufficient_common_sessions"}
    if len(sample) <= len(scope):
        return {**result, "reason": "insufficient_common_sessions_for_trial_estimate"}
    deviations = sample.std(axis=0, ddof=1)
    if np.any(deviations <= 0):
        return {**result, "reason": "constant_trial_returns"}
    correlation = np.corrcoef(sample, rowvar=False)
    rho = float(correlation[np.triu_indices(len(scope), 1)].mean())
    result.update(mean_trial_correlation=rho, correlation_matrix=correlation.tolist())
    if (
        not np.isfinite(correlation).all()
        or np.linalg.eigvalsh(correlation).min() <= 1e-12
    ):
        return {**result, "reason": "ill_conditioned_trial_correlation"}
    if not 0 <= rho < 1:
        return {**result, "reason": "effective_trial_estimate_unsupported"}
    effective = rho + (1 - rho) * len(scope)
    if effective < 2:
        return {
            **result,
            "effective_trials": effective,
            "reason": "insufficient_effective_trials",
        }
    from scipy.stats import kurtosis, norm, skew

    sharpes = sample.mean(axis=0) / deviations
    variance = float(sharpes.var(ddof=1))
    benchmark = math.sqrt(variance) * (
        (1 - np.euler_gamma) * norm.ppf(1 - 1 / effective)
        + np.euler_gamma * norm.ppf(1 - 1 / (effective * math.e))
    )
    index = scope.index(target)
    sr = float(sharpes[index])
    asymmetry = float(skew(sample[:, index], bias=False))
    fourth = float(kurtosis(sample[:, index], fisher=False, bias=False))
    denominator = 1 - asymmetry * sr + (fourth - 1) * sr * sr / 4
    if denominator <= 0:
        return {**result, "reason": "invalid_sharpe_sampling_variance"}
    probability = float(
        norm.cdf((sr - benchmark) * math.sqrt(len(sample) - 1) / math.sqrt(denominator))
    )
    return {
        **result,
        "status": "ready",
        "reason": None,
        "deflated_sharpe_probability": probability,
        "effective_trials": effective,
        "daily_sharpe": sr,
        "trial_sharpe_variance": variance,
        "expected_maximum_daily_sharpe": float(benchmark),
        "skewness": asymmetry,
        "kurtosis": fourth,
    }


def common_sample_comparison(
    baseline: pl.DataFrame, variant: pl.DataFrame, *, columns: Sequence[str]
) -> tuple[pl.DataFrame, dict]:
    """Compare saved scalar daily primitives on exactly matching finite dates."""
    for frame in (baseline, variant):
        if frame["time"].n_unique() != frame.height:
            raise ValueError("comparison requires one row per observed date")
    matched = baseline.select("time", *columns).join(
        variant.select("time", *columns), on="time", how="inner", suffix="_variant"
    )
    matched = matched.filter(
        pl.all_horizontal(
            [
                pl.col(name).is_not_null() & pl.col(name).is_finite()
                for column in columns
                for name in (column, column + "_variant")
            ]
        )
    ).sort("time")
    for column in columns:
        matched = matched.with_columns(
            (pl.col(column + "_variant") - pl.col(column)).alias(column + "_increment")
        )
    statistics = {
        column: asdict(
            hac_mean_test(matched[column + "_increment"].to_list(), window_width=1)
        )
        for column in columns
    }
    if matched.height < 12:
        for statistic in statistics.values():
            statistic.update(
                p_value=None, unavailable_reason="at_least_12_common_sessions_required"
            )
    return matched, {
        "sample_count": matched.height,
        "start": matched["time"].min(),
        "end": matched["time"].max(),
        "increments": statistics,
    }


def capacity_participation(
    fills: pl.DataFrame, turnover: pl.DataFrame, *, calendar: pl.DataFrame
) -> pl.DataFrame:
    """Participation in strictly prior 20 observed sessions' CNY traded amount.

    Turnover columns are (time, asset_id, amount); fills contain notional.
    Missing sessions never shorten the 20-session denominator.
    """
    for frame in (turnover,):
        if frame.select(pl.struct("time", "asset_id").is_duplicated().any()).item():
            raise ValueError("capacity turnover keys must be unique")
    assets = fills.select("asset_id").unique()
    dense = (
        calendar.select("time")
        .unique()
        .sort("time")
        .join(assets, how="cross")
        .join(turnover, on=["time", "asset_id"], how="left")
        .sort("asset_id", "time")
    )
    dense = dense.with_columns(
        pl.when((pl.col("amount") >= 0) & pl.col("amount").is_finite())
        .then(pl.col("amount"))
        .otherwise(None)
        .alias("amount")
    )
    dense = dense.with_columns(
        pl.col("amount")
        .shift(1)
        .rolling_mean(window_size=20, min_samples=20)
        .over("asset_id")
        .alias("adv20")
    )
    traded = fills.group_by("time", "asset_id").agg(
        pl.col("notional").abs().sum().alias("traded_notional")
    )
    return (
        traded.join(
            dense.select("time", "asset_id", "adv20"),
            on=["time", "asset_id"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("adv20") > 0)
            .then(pl.col("traded_notional") / pl.col("adv20"))
            .otherwise(None)
            .alias("participation"),
            pl.when(pl.col("adv20").is_null())
            .then(pl.lit("incomplete_prior_20_sessions"))
            .when(pl.col("adv20") <= 0)
            .then(pl.lit("zero_prior_20_session_turnover"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("unavailable_reason"),
        )
        .sort("time", "asset_id")
    )
