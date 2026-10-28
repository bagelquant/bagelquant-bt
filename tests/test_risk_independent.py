from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from bagelquant_bt import (
    evaluate_risk_factor_returns,
    fit_risk_factor_returns,
    plan_risk_factor_update,
    rolling_risk_profile,
)


def test_equal_weight_industries_are_exact_means_without_cap_inputs():
    frame = pl.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d", "e"],
            "industry": ["one", "one", "one", "two", "two"],
            "forward_return": [0.01, 0.02, 0.03, 0.05, 0.07],
        }
    )
    expected = {"market": 0.036, "industry:one": -0.016, "industry:two": 0.024}
    for data in (
        frame,
        frame.with_columns(pl.Series("market_cap", [None, 0.0, 1e12, 1.0, 100.0])),
    ):
        result = fit_risk_factor_returns(
            data,
            styles=(),
            weighting="equal",
            minimum_assets=2,
            observations_per_parameter=1,
        )
        assert result.reason is None and result.sample_count == 5
        assert result.cap_coverage is None
        actual = dict(
            result.factor_returns.select("factor", "factor_return").iter_rows()
        )
        assert actual == pytest.approx(expected)
        assert 3 * actual["industry:one"] + 2 * actual["industry:two"] == pytest.approx(
            0.0
        )


def test_equal_weight_update_has_no_cap_dependency_and_preserves_activity():
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(4)]
    exposures = pl.DataFrame(
        {
            "time": [days[0]] * 4,
            "asset_id": ["a", "b", "c", "d"],
            "industry": ["one", "one", "two", "two"],
        }
    )
    prices = pl.DataFrame(
        {
            "time": [days[1], days[2]] * 4,
            "asset_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "price": [1.0, 1.01, 1.0, 1.03, 1.0, 1.05, 1.0, 1.07],
        }
    )
    result = evaluate_risk_factor_returns(
        exposures,
        prices,
        plan_risk_factor_update(days, start=days[0], through=days[2]),
        styles=(),
        weighting="equal",
        minimum_assets=2,
        observations_per_parameter=1,
        require_continuity=True,
    )
    assert result.state["weighting"] == "equal"
    assert result.diagnostics["reason"].item() is None
    assert (
        result.industry_activity.filter(pl.col("factor") == "industry:two")[
            "sample_count"
        ].item()
        == 2
    )


def test_independent_fits_do_not_share_missing_descriptor_or_industry_inputs():
    x = np.tile([-2.0, -1.0, 1.0, 2.0], 3)
    industry = ["a"] * 8 + ["b"] * 4
    y = 0.003 + 0.002 * x + np.array([-0.01] * 8 + [0.02] * 4)
    frame = pl.DataFrame(
        {
            "asset_id": [str(i) for i in range(12)],
            "market_cap": [1.0] * 12,
            "industry": industry,
            "size": x,
            "forward_return": y,
        }
    ).with_columns(
        pl.when(pl.col("industry") == "b")
        .then(None)
        .otherwise(pl.col("size"))
        .alias("size")
    )
    fitted = fit_risk_factor_returns(
        frame, styles=["size"], minimum_assets=2, observations_per_parameter=1
    )
    assert fitted.reason is None  # An excluded industry's empty dummy is absent.
    industry_fit = fit_risk_factor_returns(
        frame.drop("size"), styles=(), minimum_assets=2, observations_per_parameter=1
    )
    assert dict(
        industry_fit.factor_returns.select("factor", "factor_return").iter_rows()
    ) == pytest.approx({"market": 0.003, "industry:a": -0.01, "industry:b": 0.02})
    style = fit_risk_factor_returns(
        frame.drop("industry"),
        styles=["size"],
        industry_controls=False,
        minimum_assets=2,
        observations_per_parameter=1,
    )
    assert dict(
        style.factor_returns.select("factor", "factor_return").iter_rows()
    ) == pytest.approx({"market": -0.007, "size": 0.002})
    assert style.parameter_count == style.rank == 2


def test_industry_update_needs_no_styles_and_records_members_without_observations():
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(4)]
    exposures = pl.DataFrame(
        {
            "time": [days[0]] * 4,
            "asset_id": ["a", "b", "c", "d"],
            "market_cap": [1.0] * 4,
            "industry": ["one", "one", "two", "two"],
        }
    )
    prices = pl.DataFrame(
        {
            "time": [days[1], days[2]] * 2,
            "asset_id": ["a", "a", "b", "b"],
            "price": [1.0, 0.99, 1.0, 1.01],
        }
    )
    result = evaluate_risk_factor_returns(
        exposures,
        prices,
        plan_risk_factor_update(days[:3], start=days[0], through=days[2]),
        styles=(),
        minimum_assets=2,
        observations_per_parameter=1,
    )
    assert result.diagnostics["reason"][0] == "industry_observations_missing"
    missing = result.industry_activity.filter(pl.col("factor") == "industry:two").row(
        0, named=True
    )
    assert (
        missing["active"]
        and missing["eligible_count"] == 2
        and missing["sample_count"] == 0
    )


def test_strict_profile_rejects_interior_gap_without_moving_its_start():
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(32)]
    x = np.sin(np.arange(31)) / 100
    gross = pl.DataFrame({"time": days[:-1], "gross_return": 0.001 + 2 * x})
    factors = pl.DataFrame(
        {
            "time": days[:-1],
            "available_date": days[1:],
            "factor": ["size"] * 31,
            "family": ["style"] * 31,
            "factor_return": x,
        }
    )
    good = rolling_risk_profile(
        gross,
        factors,
        days,
        regression_window=10,
        regression_min_periods=4,
        volatility_window=4,
        require_continuity=True,
    )
    assert good.state["valid_start"] == str(days[4])
    for broken in [
        factors.filter(pl.col("time") != days[20]),
        factors.with_columns(
            pl.when(pl.col("time") == days[20])
            .then(None)
            .otherwise(pl.col("factor_return"))
            .alias("factor_return")
        ),
    ]:
        with pytest.raises(ValueError, match=r"RISK_CONTINUITY_REQUIRED.*2020-01-21"):
            rolling_risk_profile(
                gross,
                broken,
                days,
                regression_window=10,
                regression_min_periods=4,
                volatility_window=4,
                require_continuity=True,
            )


def test_masked_industry_regression_uses_actual_active_observations():
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(33)]
    rng = np.random.default_rng(14)
    market, sector = rng.normal(0, 0.01, (2, 32))
    active = np.array([i < 12 or i >= 20 for i in range(32)])
    rows, activity = [], []
    for i in range(32):
        for name, family, value in [
            ("market", "market", market[i]),
            *([("industry:a", "industry", sector[i])] if active[i] else []),
        ]:
            rows.append(
                {
                    "time": days[i],
                    "available_date": days[i + 1],
                    "factor": name,
                    "family": family,
                    "factor_return": value,
                }
            )
            activity.append(
                {
                    "time": days[i],
                    "available_date": days[i + 1],
                    "factor": name,
                    "active": True,
                }
            )
    gross_values = 0.001 + 1.5 * market + 2 * np.where(active, sector, 0)
    result = rolling_risk_profile(
        pl.DataFrame({"time": days[:-1], "gross_return": gross_values}),
        pl.DataFrame(rows),
        days,
        industry_activity=pl.DataFrame(activity),
        regression_window=10,
        regression_min_periods=4,
        volatility_window=4,
        require_continuity=True,
    )
    # Independent reference: the declared inactive model term is absent.
    design = np.column_stack([np.where(active[11:21], sector[11:21], 0), market[11:21]])
    scale = design.std(axis=0)
    z = (design - design.mean(axis=0)) / scale
    y = gross_values[11:21]
    expected = np.linalg.solve(z.T @ z + np.eye(2), z.T @ (y - y.mean())) / scale
    actual = result.exposures.filter(pl.col("time") == days[21]).sort("factor")
    assert actual["beta"].to_numpy() == pytest.approx(expected)
    assert result.exposures.filter(pl.col("time").is_in(days[12:20]))[
        "factor"
    ].unique().to_list() == ["market"]


def test_strict_mature_append_rejects_empty_gross(monkeypatch):
    from test_risk_profile_incremental import SETTINGS, evaluate, inputs

    monkeypatch.setitem(SETTINGS, "require_continuity", True)
    days, returns, factors = inputs()
    original = evaluate(days, returns, factors, days[17])
    with pytest.raises(
        ValueError, match=r"RISK_CONTINUITY_REQUIRED.*return_unavailable"
    ):
        evaluate(days, returns.clear(), factors, days[18], previous=original)


def test_activity_cannot_hide_an_actual_industry_return():
    from test_risk_profile_incremental import inputs

    days, returns, factors = inputs()
    industry = factors.with_columns(
        pl.lit("industry:a").alias("factor"), pl.lit("industry").alias("family")
    )
    market = factors.with_columns(
        pl.lit("market").alias("factor"), pl.lit("market").alias("family")
    )
    activity = (
        pl.concat([market, industry])
        .select("time", "available_date", "factor")
        .with_columns(pl.lit(True).alias("active"))
    )
    broken = activity.filter(
        ~((pl.col("time") == days[20]) & (pl.col("factor") == "industry:a"))
    )
    with pytest.raises(ValueError, match=r"contradicts.*activity"):
        rolling_risk_profile(
            returns,
            pl.concat([market, industry]),
            days,
            industry_activity=broken,
            require_continuity=True,
            regression_window=5,
            volatility_window=3,
        )


def test_nonfinite_fitted_coefficients_cannot_be_ready():
    frame = pl.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"],
            "market_cap": [1.0] * 4,
            "size": [1e-8, 2e-8, 3e-8, 4e-8],
            "forward_return": [0.0, 0.0, 1e308, 1e308],
        }
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result = fit_risk_factor_returns(
            frame,
            styles=["size"],
            industry_controls=False,
            minimum_assets=1,
            observations_per_parameter=1,
        )
    assert result.reason == "nonfinite_fit"
    assert (
        result.factor_returns["factor_return"].null_count()
        == result.factor_returns.height
    )


def test_strict_fit_repair_keeps_original_warmup_and_start():
    from test_risk_update import inputs, run

    days, exposures, prices = inputs()
    warmup = exposures.with_columns(
        pl.when(pl.col("time") < days[3])
        .then(None)
        .otherwise(pl.col("size"))
        .alias("size")
    )
    plan = plan_risk_factor_update(days, start=days[0], through=days[-1])
    original = run(warmup, prices, plan, require_continuity=True)
    start = days[4]
    assert original.state["valid_start"] == str(start)
    assert original.state["fit_coverage"] == {
        "times": [str(day) for day in days[1:-1]],
        "first_valid": str(start),
        "unavailable": [
            {"time": str(day), "reason": "insufficient_cross_section"}
            for day in days[1:4]
        ],
    }
    repaired = run(warmup, prices, plan, require_continuity=True, valid_start=start)
    assert repaired.state["valid_start"] == str(start)
    broken = warmup.with_columns(
        pl.when(pl.col("time") == days[6])
        .then(None)
        .otherwise(pl.col("size"))
        .alias("size")
    )
    with pytest.raises(ValueError, match="RISK_CONTINUITY_REQUIRED"):
        run(broken, prices, plan, require_continuity=True, valid_start=start)
    with pytest.raises(ValueError, match="RISK_START_REPAIR_REQUIRED"):
        run(exposures, prices, plan, require_continuity=True, valid_start=start)
