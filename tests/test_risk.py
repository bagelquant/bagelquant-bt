from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from bagelquant_bt import (
    fit_risk_factor_returns,
    link_risk_contributions,
    rolling_risk_profile,
    standardize_risk_descriptor,
)


def test_constrained_wls_recovers_known_returns():
    rng = np.random.default_rng(18)
    n = 240
    caps = np.arange(1, n + 1, dtype=float)
    industry = np.where(np.arange(n) < 100, "a", "b")
    x = rng.normal(size=n)
    industry_a = 0.003
    industry_b = -caps[:100].sum() / caps[100:].sum() * industry_a
    y = 0.002 + np.where(industry == "a", industry_a, industry_b) + 0.01 * x
    fit = fit_risk_factor_returns(
        pl.DataFrame(
            {
                "asset_id": [str(i) for i in range(n)],
                "industry": industry,
                "market_cap": caps,
                "size": x,
                "forward_return": y,
            }
        ),
        styles=["size"],
    )
    assert fit.reason is None
    actual = dict(fit.factor_returns.select("factor", "factor_return").iter_rows())
    assert actual == pytest.approx(
        {
            "market": 0.002,
            "industry:a": industry_a,
            "industry:b": industry_b,
            "size": 0.01,
        }
    )
    assert actual["industry:a"] * caps[:100].sum() + actual["industry:b"] * caps[
        100:
    ].sum() == pytest.approx(0)


def _profile_inputs():
    sessions = [date(2020, 1, 1) + timedelta(days=i) for i in range(361)]
    rng = np.random.default_rng(8)
    x = rng.normal(0, 0.01, 360)
    returns = pl.DataFrame({"time": sessions[:-1], "gross_return": 0.001 + 2 * x})
    factors = pl.DataFrame(
        {
            "time": sessions[:-1],
            "available_date": sessions[1:],
            "factor": ["size"] * 360,
            "family": ["style"] * 360,
            "factor_return": x,
        }
    )
    return sessions, returns, factors


def test_ridge_is_lagged_and_coefficients_are_in_original_units():
    days, returns, factors = _profile_inputs()
    profile = rolling_risk_profile(returns, factors, days)
    beta = profile.exposures["beta"].to_numpy()
    # Centered standardized predictor has sum of squares 240; fixed penalty is 1.
    assert beta == pytest.approx(np.full(120, 2 * 240 / 241))
    assert profile.returns["specific_volatility"].drop_nulls().len() == 1
    changed = returns.with_columns(
        pl.when(pl.col("time") >= days[280])
        .then(100.0)
        .otherwise(pl.col("gross_return"))
        .alias("gross_return")
    )
    later = rolling_risk_profile(changed, factors, days)
    assert later.exposures.filter(pl.col("time") <= days[280]).equals(
        profile.exposures.filter(pl.col("time") <= days[280])
    )
    assert later.returns.filter(pl.col("time") < days[280]).equals(
        profile.returns.filter(pl.col("time") < days[280])
    )


def test_missing_training_day_is_not_compressed_and_late_labels_are_rejected():
    days, returns, factors = _profile_inputs()
    missing = factors.with_columns(
        pl.when(pl.col("time") == days[10])
        .then(None)
        .otherwise(pl.col("factor_return"))
        .alias("factor_return")
    )
    result = rolling_risk_profile(returns, missing, days)
    assert result.exposures["time"].min() == days[251]
    delayed = factors.with_columns(
        pl.when(pl.col("time") == days[239])
        .then(days[241])
        .otherwise(pl.col("available_date"))
        .alias("available_date")
    )
    result = rolling_risk_profile(returns, delayed, days)
    assert (
        result.diagnostics.filter(pl.col("time") == days[240])["reason"][0]
        == "incomplete_training_window"
    )


def test_linked_contributions_reconcile_and_rebase():
    days, returns, factors = _profile_inputs()
    result = rolling_risk_profile(returns, factors, days)
    linked = link_risk_contributions(result.returns, result.exposures)
    sums = (
        linked.filter(pl.col("factor") != "alpha")
        .group_by("time")
        .agg(pl.col("cumulative_contribution").sum().alias("sum"))
    )
    alpha = (
        linked.filter(pl.col("factor") == "alpha").join(sums, on="time").drop_nulls()
    )
    assert alpha["cumulative_contribution"].to_numpy() == pytest.approx(
        alpha["sum"].to_numpy()
    )
    clipped = link_risk_contributions(
        result.returns.filter(pl.col("time") >= days[300]),
        result.exposures.filter(pl.col("time") >= days[300]),
    )
    first = clipped.filter(
        (pl.col("time") == days[300]) & (pl.col("factor") == "alpha")
    )
    assert first["cumulative_contribution"][0] == pytest.approx(
        returns.filter(pl.col("time") == days[300])["gross_return"][0]
    )


def test_null_breaks_linked_path_and_constant_descriptor_is_unavailable():
    days, returns, factors = _profile_inputs()
    result = rolling_risk_profile(returns, factors, days)
    broken = result.returns.with_columns(
        pl.when(pl.col("time") == days[300])
        .then(None)
        .otherwise(pl.col("specific_return"))
        .alias("specific_return")
    )
    linked = link_risk_contributions(broken, result.exposures)
    next_alpha = linked.filter(
        (pl.col("time") == days[301]) & (pl.col("factor") == "alpha")
    )
    assert next_alpha["cumulative_contribution"][0] == pytest.approx(
        returns.filter(pl.col("time") == days[301])["gross_return"][0]
    )
    assert np.isnan(standardize_risk_descriptor(np.ones(5))).all()


def test_rank_deficiency_is_reported_without_fabricated_factor_returns():
    frame = pl.DataFrame(
        {
            "asset_id": [str(i) for i in range(220)],
            "industry": ["a"] * 220,
            "market_cap": [1.0] * 220,
            "size": [1.0] * 220,
            "forward_return": [0.01] * 220,
        }
    )
    result = fit_risk_factor_returns(frame, styles=["size"])
    assert result.reason == "rank_deficient"
    assert result.factor_returns["factor_return"].null_count() == 3


def test_explicit_calendar_freezes_market_wide_gap_without_early_recovery():
    from bagelquant_bt import risk_interval_returns

    days = [date(2024, 1, i) for i in range(1, 6)]
    prices = pl.DataFrame(
        {
            "time": [days[0], days[3], days[4]],
            "asset_id": ["a"] * 3,
            "price": [10.0, 12.0, 15.0],
        }
    )
    calendar = pl.DataFrame({"time": days})
    result = risk_interval_returns(prices, calendar)
    assert result["forward_return"].to_list() == pytest.approx([0, 0, 0.2, 0.25])
    assert result["available_date"].to_list() == days[1:]
    before = risk_interval_returns(
        prices.filter(pl.col("time") <= days[2]),
        calendar.filter(pl.col("time") <= days[2]),
    )
    assert result.filter(pl.col("available_date") <= days[2]).equals(before)
    resumed = risk_interval_returns(prices, calendar.filter(pl.col("time") >= days[2]))
    assert resumed["forward_return"][0] == pytest.approx(0.2)
