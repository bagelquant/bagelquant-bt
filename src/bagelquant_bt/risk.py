"""Explicit, capital-free factor estimation and causal return attribution.

Return ``time`` is the start of an interval; ``available_date`` is its end.
These analytics accept numerical inputs, never resolve data or create accounts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from .inputs import validate_prices
from .risk_profile_update import (
    RiskProfileUpdatePlan,
    plan_risk_profile_update,
    risk_profile_policy,
)


def risk_interval_returns(prices: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    """Calendar-aligned open-to-open valuation, freezing gaps until recovery.

    The bounded input may include one earlier observation per asset. Even a
    market-wide price gap must not compress the trading calendar or mature a
    future recovery return early.
    """
    prices = validate_prices(prices)
    sessions = calendar.select("time").unique().sort("time")
    grid = sessions.join(prices.select("asset_id").unique(), how="cross")
    valued = grid.sort(["asset_id", "time"]).join_asof(
        prices.sort(["asset_id", "time"]),
        on="time",
        by="asset_id",
        strategy="backward",
        check_sortedness=False,
    )
    return (
        valued.with_columns(
            (pl.col("price").shift(-1).over("asset_id") / pl.col("price") - 1).alias(
                "forward_return"
            ),
            pl.col("time").shift(-1).over("asset_id").alias("available_date"),
        )
        .drop_nulls("available_date")
        .select("time", "asset_id", "forward_return", "available_date")
        .sort(["time", "asset_id"])
    )


@dataclass(frozen=True)
class CrossSectionalRiskFit:
    factor_returns: pl.DataFrame
    sample_count: int
    eligible_count: int
    excluded_count: int
    cap_coverage: float | None
    rank: int
    parameter_count: int
    condition_number: float | None
    reason: str | None


def fit_risk_factor_returns(
    frame: pl.DataFrame,
    *,
    styles: Sequence[str],
    minimum_assets: int = 200,
    observations_per_parameter: int = 5,
) -> CrossSectionalRiskFit:
    """Fit country + industry + styles with a cap-constrained industry basis.

    Inputs: asset_id, industry, market_cap, forward_return, and style columns.
    WLS minimizes sum(normalized sqrt(cap) * residual**2). The industry
    coefficients satisfy sum(industry_cap_share * industry_return) == 0.
    """
    required = {"asset_id", "industry", "market_cap", "forward_return", *styles}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"missing risk inputs: {sorted(required - set(frame.columns))}"
        )
    if frame.get_column("asset_id").n_unique() != frame.height:
        raise ValueError("risk estimation requires unique asset_id")
    numeric = ["market_cap", "forward_return", *styles]
    finite = pl.all_horizontal(
        [pl.col(c).is_finite().fill_null(False) for c in numeric]
    )
    sample = frame.filter(
        finite
        & (pl.col("market_cap") > 0)
        & pl.col("industry").is_not_null()
        & (pl.col("industry").cast(pl.String).str.len_chars() > 0)
    ).sort("asset_id")
    industries = sorted(
        frame.filter(
            pl.col("market_cap").is_finite()
            & (pl.col("market_cap") > 0)
            & pl.col("industry").is_not_null()
            & (pl.col("industry").cast(pl.String).str.len_chars() > 0)
        )
        .get_column("industry")
        .cast(pl.String)
        .unique()
        .to_list()
    )
    names = ["market", *(f"industry:{i}" for i in industries), *styles]
    families = ["market", *(["industry"] * len(industries)), *(["style"] * len(styles))]
    p = 1 + max(0, len(industries) - 1) + len(styles)
    caps = sample.get_column("market_cap").to_numpy()
    total_cap = (
        frame.filter(pl.col("market_cap").is_finite() & (pl.col("market_cap") > 0))
        .get_column("market_cap")
        .sum()
    )
    coverage = float(caps.sum() / total_cap) if total_cap else None
    rank, condition, reason = 0, None, None
    coefficients = np.full(len(names), np.nan)
    if sample.height < max(minimum_assets, observations_per_parameter * p):
        reason = "insufficient_cross_section"
    elif not industries:
        reason = "industry_missing"
    else:
        industry = sample.get_column("industry").cast(pl.String).to_numpy()
        dummies = np.column_stack([industry == i for i in industries]).astype(float)
        industry_caps = dummies.T @ caps
        # Eliminate the largest industry's coefficient for numerical stability.
        reference = int(np.argmax(industry_caps))
        others = [i for i in range(len(industries)) if i != reference]
        contrast = dummies[:, others] - dummies[:, [reference]] * (
            industry_caps[others] / industry_caps[reference]
        )
        x = np.column_stack(
            [np.ones(sample.height), contrast, sample.select(styles).to_numpy()]
        )
        weights = np.sqrt(caps)
        weights /= weights.sum()
        a = x * np.sqrt(weights)[:, None]
        y = sample.get_column("forward_return").to_numpy()
        b, _, rank, singular = np.linalg.lstsq(a, y * np.sqrt(weights), rcond=None)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
        if rank != p:
            reason = "rank_deficient"
        else:
            industry_b = np.zeros(len(industries))
            industry_b[others] = b[1 : len(industries)]
            industry_b[reference] = (
                -float(industry_caps[others] @ industry_b[others])
                / industry_caps[reference]
            )
            coefficients = np.concatenate([[b[0]], industry_b, b[len(industries) :]])
    factors = pl.DataFrame(
        {"factor": names, "family": families, "factor_return": coefficients}
    ).with_columns(pl.col("factor_return").fill_nan(None))
    return CrossSectionalRiskFit(
        factors,
        sample.height,
        frame.height,
        frame.height - sample.height,
        coverage,
        int(rank),
        p,
        condition,
        reason,
    )


@dataclass(frozen=True)
class RollingRiskProfile:
    returns: pl.DataFrame
    exposures: pl.DataFrame
    diagnostics: pl.DataFrame
    state: dict[str, Any] = field(default_factory=dict)
    solve_count: int = 0


def rolling_risk_profile(
    returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
    calendar: Sequence[date],
    *,
    regression_window: int = 240,
    volatility_window: int = 120,
    annualization: int = 240,
    ridge: float = 1.0,
    previous: RollingRiskProfile | None = None,
    plan: RiskProfileUpdatePlan | None = None,
    output_start: date | None = None,
    input_proof: Callable[[date, date, date | None], str | None] | None = None,
    previous_input_proof: Callable[[date, date, date | None], str | None] | None = None,
    factor_catalogue: Mapping[str, Mapping[str, str]] | None = None,
    check_canceled: Callable[[], None] | None = None,
) -> RollingRiskProfile:
    """Attribute one Gross path using only previously matured observations.

    ``returns`` has time/gross_return. Factors have time/available_date/factor/
    family/factor_return. Calendar gaps are never compressed into valid rows.
    A factor absent before its first historical appearance is inactive (zero);
    an explicit null factor return always means unavailable. Factors enter the
    output population only at their first causal appearance (policy v2).

    A previous result needs its compact original state and only the saved
    specific-return halo, not historical regressions/exposures. Gross/factor
    inputs may be restricted to ``plan.read_start``. Continuation emits only the
    planned output delta; publish it with the original BT-owned references.
    ``factor_catalogue`` supplies original family/first_seen metadata when a
    fresh diagnostic starts after the factor history. It does not supply any
    pre-start regression observations or advance a factor's causal appearance.
    """
    policy = risk_profile_policy(
        regression_window, volatility_window, annualization, ridge
    )
    cancel = check_canceled or (lambda: None)
    cancel()
    if (
        returns.schema.get("time") != pl.Date
        or factor_returns.schema.get("time") != pl.Date
    ):
        raise ValueError("risk profile inputs require Date interval coordinates")
    if returns.get_column("time").n_unique() != returns.height:
        raise ValueError("duplicate return interval")
    if factor_returns.unique(["time", "factor"]).height != factor_returns.height:
        raise ValueError("duplicate factor interval")
    if len(calendar) == 0:
        raise ValueError("risk profile needs a nonempty calendar")
    if plan is None:
        plan = plan_risk_profile_update(
            None if previous is None else previous.state,
            calendar=calendar,
            through=calendar[-1],
            input_proof=input_proof,
            regression_window=regression_window,
            volatility_window=volatility_window,
            annualization=annualization,
            ridge=ridge,
        )
    if plan.policy != policy or tuple(calendar) != plan.calendar:
        raise ValueError("risk profile plan differs from its numerical policy/calendar")
    if plan.mode == "blocked":
        raise ValueError(plan.reason)
    if previous is not None:
        state = previous.state
        proof = previous_input_proof or input_proof
        if (
            state.get("policy") != policy
            or proof is None
            or state.get("input_prefix") is None
            or input_proof is None
            or input_proof(
                date.fromisoformat(state["calendar_start"]),
                date.fromisoformat(state["proof_through"]),
                date.fromisoformat(state["cutoff"]),
            )
            != state["input_prefix"]
            or proof(
                date.fromisoformat(state["calendar_start"]),
                date.fromisoformat(state["proof_through"]),
                date.fromisoformat(state["cutoff"]),
            )
            != state["input_prefix"]
        ):
            raise ValueError(
                "risk profile continuation needs original policy/state/"
                "source-prefix proof"
            )
    if plan.complete:
        if previous is None:
            raise ValueError("risk profile completed plan needs its original state")
        return _risk_profile_result([], [], [], dict(previous.state), 0)
    if plan.output_start is None or plan.read_start is None or plan.output_end is None:
        raise ValueError(
            "risk profile execution requires an admitted output/input window"
        )
    if output_start is not None and output_start != plan.output_start:
        raise ValueError("risk profile output start differs from its admitted plan")
    if previous is None and plan.mode != "initialize":
        raise ValueError(
            "risk profile continuation needs the selected original checkpoint"
        )
    admitted = [day for day in calendar if day <= plan.through]
    global_positions = {day: i for i, day in enumerate(admitted)}
    if plan.mode == "repair":
        before = global_positions[plan.output_start] - 1
        if (
            before < 0
            or previous_input_proof is None
            or input_proof(admitted[0], admitted[before], plan.through) is None
            or input_proof(admitted[0], admitted[before], plan.through)
            != previous_input_proof(admitted[0], admitted[before], plan.through)
        ):
            raise ValueError(
                "risk profile repair needs original source equality for its saved halo"
            )
    first_position = global_positions[plan.read_start]
    last_position = global_positions[plan.output_end]
    sessions = admitted[first_position : min(last_position + 2, len(admitted))]
    output_position = global_positions[plan.output_start] - first_position
    rows = {
        r["time"]: r
        for r in returns.filter(
            pl.col("time").is_between(plan.read_start, plan.output_end)
        ).iter_rows(named=True)
    }
    factor_returns = factor_returns.filter(
        pl.col("time").is_between(plan.read_start, plan.output_end)
    )
    retained_state = plan.retained_state or (
        None if previous is None else previous.state
    )
    catalog = (
        {}
        if retained_state is None
        else {name: dict(item) for name, item in retained_state["factors"].items()}
    )
    for name, item in (factor_catalogue or {}).items():
        first_seen = date.fromisoformat(item["first_seen"])
        if first_seen > plan.through:
            continue
        if name in catalog and catalog[name] != dict(item):
            raise ValueError(
                "risk factor catalogue changed; explicitly plan its repair"
            )
        catalog[name] = {"family": item["family"], "first_seen": str(first_seen)}
    for name, family, first_seen_date in (
        factor_returns.group_by("factor")
        .agg(
            pl.col("family").n_unique().alias("family_count"),
            pl.col("family").first(),
            pl.col("time").min().alias("first_seen"),
        )
        .filter(pl.col("family_count") == 1)
        .select("factor", "family", "first_seen")
        .iter_rows()
    ):
        if name in (factor_catalogue or {}) and first_seen_date < date.fromisoformat(
            factor_catalogue[name]["first_seen"]
        ):
            raise ValueError("risk factor observation precedes its original catalogue")
        if name in catalog:
            if catalog[name]["family"] != family:
                raise ValueError(
                    "risk factor family changed; explicitly rebuild "
                    "its numerical definition"
                )
            catalog[name]["first_seen"] = min(
                catalog[name]["first_seen"], str(first_seen_date)
            )
        else:
            catalog[name] = {"family": family, "first_seen": str(first_seen_date)}
    if (
        factor_returns["factor"].n_unique()
        != factor_returns.select("factor", "family").unique().height
    ):
        raise ValueError("risk factors require one immutable family per factor")
    if (
        plan.mode == "repair"
        and plan.output_end < admitted[-1]
        and catalog != retained_state["factors"]
    ):
        raise ValueError(
            "risk factor population changed; explicitly plan "
            "its complete affected suffix"
        )
    factor_names = sorted(catalog)
    factor_index = {name: i for i, name in enumerate(factor_names)}
    families = {name: item["family"] for name, item in catalog.items()}
    first_seen = {
        name: date.fromisoformat(item["first_seen"]) for name, item in catalog.items()
    }
    by_time: dict[date, dict[str, dict]] = {}
    for row in factor_returns.iter_rows(named=True):
        by_time.setdefault(row["time"], {})[row["factor"]] = row
    n, k = len(sessions), len(factor_names)
    x, y = np.full((n, k), np.nan), np.full(n, np.nan)
    available = [None] * n
    for i, day in enumerate(sessions):
        value = rows.get(day, {}).get("gross_return")
        if value is not None:
            y[i] = value
        factors = by_time.get(day, {})
        for name, j in factor_index.items():
            item = factors.get(name)
            if item is not None:
                value = item["factor_return"]
                x[i, j] = value if value is not None else np.nan
            elif families[name] == "industry" and day < first_seen[name] and factors:
                x[i, j] = 0.0
        dates = [
            r["available_date"]
            for r in factors.values()
            if r["available_date"] is not None
        ]
        if dates and len(set(dates)) == 1 and len(dates) == len(factors):
            available[i] = max(dates)
    specific = np.full(n, np.nan)
    if previous is not None:
        origin = previous.state.get("output_origin")
        expected = {
            day
            for day in admitted
            if plan.specific_start <= day < plan.output_start
            and origin is not None
            and day >= date.fromisoformat(origin)
        }
        retained = previous.returns.filter(pl.col("time").is_in(sorted(expected)))
        if (
            retained["time"].n_unique() != retained.height
            or set(retained["time"]) != expected
        ):
            raise ValueError(
                "risk profile specific-return halo is incomplete; "
                "restore original state"
            )
        local_positions = {day: i for i, day in enumerate(sessions)}
        for day, value in retained.select("time", "specific_return").iter_rows():
            if day in local_positions and value is not None:
                specific[local_positions[day]] = value
    output, exposures, diagnostics = [], [], []
    first, last = (min(rows), max(rows)) if rows else (None, None)
    solves = 0
    for i, day in enumerate(sessions):
        cancel()
        if (
            i < output_position
            or day > plan.output_end
            or first is None
            or day < first
            or day > last
        ):
            continue
        visible = [j for j, name in enumerate(factor_names) if first_seen[name] <= day]
        active_names = [factor_names[j] for j in visible]
        active_count = len(visible)
        reason, r_squared = None, None
        start = max(0, i - regression_window)
        history_x, history_y = x[start:i, :][:, visible], y[start:i]
        current_x = x[i, visible]
        valid = np.isfinite(history_x).all(axis=1) & np.isfinite(history_y)
        matured = np.array(
            [
                d is not None and d <= day and d == sessions[start + offset + 1]
                for offset, d in enumerate(available[start:i])
            ],
            dtype=bool,
        )
        sample_count = int((valid & matured).sum()) if active_count else 0
        constant_count = 0
        if first_position + i < regression_window:
            reason = "regression_warmup"
        elif not active_count or not np.isfinite(current_x).all():
            reason = "factor_return_unavailable"
        elif not np.isfinite(y[i]) or i + 1 >= n:
            reason = "return_unavailable"
        else:
            if sample_count != regression_window:
                reason = "incomplete_training_window"
            elif available[i] is None or available[i] != sessions[i + 1]:
                reason = "return_interval_mismatch"
            else:
                mean, scale = history_x.mean(axis=0), history_x.std(axis=0, ddof=0)
                identifiable = scale > 1e-12
                constant_count = int((~identifiable).sum())
                z = np.zeros_like(history_x)
                z[:, identifiable] = (
                    history_x[:, identifiable] - mean[identifiable]
                ) / scale[identifiable]
                centered_y = history_y - history_y.mean()
                cancel()
                solves += 1
                fitted = np.linalg.solve(
                    z.T @ z + ridge * np.eye(active_count), z.T @ centered_y
                )
                beta = np.zeros(active_count)
                beta[identifiable] = fitted[identifiable] / scale[identifiable]
                residual = centered_y - z @ fitted
                denominator = float(centered_y @ centered_y)
                r_squared = (
                    1 - float(residual @ residual) / denominator
                    if denominator > 0
                    else None
                )
                contributions = beta * current_x
                specific[i] = y[i] - contributions.sum()
                for j, name in enumerate(active_names):
                    exposures.append(
                        {
                            "time": day,
                            "factor": name,
                            "family": families[name],
                            "beta": float(beta[j]),
                            "contribution": float(contributions[j]),
                            "identifiable": bool(identifiable[j]),
                        }
                    )

        def volatility(values: np.ndarray, i: int = i) -> float | None:
            if first_position + i + 1 < volatility_window or i + 1 < volatility_window:
                return None
            sample = values[i + 1 - volatility_window : i + 1]
            return (
                float(sample.std(ddof=1) * np.sqrt(annualization))
                if np.isfinite(sample).all()
                else None
            )

        output.append(
            {
                "time": day,
                "gross_return": float(y[i]) if np.isfinite(y[i]) else None,
                "specific_return": float(specific[i])
                if np.isfinite(specific[i])
                else None,
                "gross_volatility": volatility(y),
                "specific_volatility": volatility(specific),
            }
        )
        diagnostics.append(
            {
                "time": day,
                "reason": reason,
                "sample_count": sample_count,
                "training_coverage": sample_count / regression_window,
                "r_squared": r_squared,
                "constant_factor_count": constant_count,
                "regression_window": regression_window,
                "ridge": ridge,
            }
        )
    complete_positions = [
        i
        for i, day in enumerate(sessions[:-1])
        if np.isfinite(y[i])
        and any(first_seen[name] <= day for name in factor_names)
        and np.isfinite(
            x[i, [j for j, name in enumerate(factor_names) if first_seen[name] <= day]]
        ).all()
        and available[i] == sessions[i + 1]
    ]
    pending_position = (
        complete_positions[-1] + 1 if complete_positions else output_position
    )
    origin = None if first is None else str(first)
    if previous is not None and previous.state.get("output_origin"):
        origin = min(
            origin or previous.state["output_origin"], previous.state["output_origin"]
        )
    proof_through = sessions[pending_position - 1] if pending_position else admitted[0]
    if plan.mode == "repair" and plan.output_end < admitted[-1]:
        proof_through = date.fromisoformat(retained_state["proof_through"])
    state = {
        "policy": policy,
        "calendar_start": str(admitted[0]),
        "through": str(admitted[-1]),
        "cutoff": str(plan.through),
        "output_origin": origin,
        "pending_output_start": str(sessions[pending_position]),
        "factors": catalog,
        "proof_through": str(proof_through),
        "input_prefix": None
        if input_proof is None
        else input_proof(admitted[0], proof_through, plan.through),
    }
    if plan.mode == "repair" and plan.output_end < admitted[-1]:
        state["pending_output_start"] = retained_state["pending_output_start"]
        state["output_origin"] = retained_state["output_origin"]
    cancel()
    return _risk_profile_result(output, exposures, diagnostics, state, solves)


def _risk_profile_result(output, exposures, diagnostics, state, solves):
    return RollingRiskProfile(
        pl.DataFrame(
            output,
            schema={
                "time": pl.Date,
                "gross_return": pl.Float64,
                "specific_return": pl.Float64,
                "gross_volatility": pl.Float64,
                "specific_volatility": pl.Float64,
            },
        ),
        pl.DataFrame(
            exposures,
            schema={
                "time": pl.Date,
                "factor": pl.String,
                "family": pl.String,
                "beta": pl.Float64,
                "contribution": pl.Float64,
                "identifiable": pl.Boolean,
            },
        ),
        pl.DataFrame(
            diagnostics,
            schema={
                "time": pl.Date,
                "reason": pl.String,
                "sample_count": pl.Int64,
                "training_coverage": pl.Float64,
                "r_squared": pl.Float64,
                "constant_factor_count": pl.Int64,
                "regression_window": pl.Int64,
                "ridge": pl.Float64,
            },
        ),
        state,
        solves,
    )


def industry_exposure_strength(exposures: pl.DataFrame) -> pl.DataFrame:
    """Return the L2 norm of the complete industry-beta vector on each date.

    Callers select one return metric before invoking this function. Structural
    zero betas are valid; missing or non-finite coordinates invalidate the date.
    The norm measures exposure magnitude, not direction or variance contribution.
    """
    schema = {"time": pl.Date, "beta": pl.Float64}
    if exposures.is_empty():
        return pl.DataFrame(schema=schema)
    industry = exposures.filter(pl.col("family") == "industry")
    if industry.is_empty():
        return pl.DataFrame(schema=schema)
    if industry.select("time", "factor").unique().height != industry.height:
        raise ValueError("industry exposure requires unique time/factor coordinates")
    expected = industry["factor"].n_unique()
    return (
        industry.group_by("time")
        .agg(
            pl.when(
                (pl.len() == expected)
                & pl.col("beta").is_finite().fill_null(False).all()
            )
            .then(pl.col("beta").pow(2).sum().sqrt())
            .otherwise(None)
            .alias("beta")
        )
        .sort("time")
    )


def link_risk_contributions(
    returns: pl.DataFrame, exposures: pl.DataFrame
) -> pl.DataFrame:
    """Rebase each contiguous valid segment; never bridge an unavailable date.

    The returned contribution series add exactly to the compounded Gross path.
    Specific's independent compounded path is a separate column.
    """
    contributions: dict[date, dict[str, float]] = {}
    for row in exposures.iter_rows(named=True):
        key = "industry" if row["family"] == "industry" else row["factor"]
        day = contributions.setdefault(row["time"], {})
        day[key] = day.get(key, 0.0) + row["contribution"]
    wealth, residual_wealth, linked, output, segment = 1.0, 1.0, {}, [], 0
    for row in returns.sort("time").iter_rows(named=True):
        gross, specific = row["gross_return"], row["specific_return"]
        if (
            gross is None
            or specific is None
            or not np.isfinite([gross, specific]).all()
        ):
            wealth, residual_wealth, linked = 1.0, 1.0, {}
            segment += 1
            output.append(
                {
                    "time": row["time"],
                    "factor": "alpha",
                    "segment": segment,
                    "cumulative_contribution": None,
                    "specific_cumulative_return": None,
                }
            )
            continue
        daily = {**contributions.get(row["time"], {}), "specific": specific}
        if not np.isclose(sum(daily.values()), gross, rtol=1e-9, atol=1e-12):
            raise ValueError("risk contributions do not reconcile to Gross return")
        for name, value in daily.items():
            linked[name] = linked.get(name, 0.0) + wealth * value
        wealth *= 1 + gross
        residual_wealth *= 1 + specific
        for name, value in {**linked, "alpha": wealth - 1}.items():
            output.append(
                {
                    "time": row["time"],
                    "factor": name,
                    "segment": segment,
                    "cumulative_contribution": value,
                    "specific_cumulative_return": residual_wealth - 1
                    if name == "specific"
                    else None,
                }
            )
    return pl.DataFrame(
        output,
        schema={
            "time": pl.Date,
            "factor": pl.String,
            "segment": pl.Int64,
            "cumulative_contribution": pl.Float64,
            "specific_cumulative_return": pl.Float64,
        },
    )
