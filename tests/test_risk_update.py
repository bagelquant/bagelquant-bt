from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt import (
    RISK_UPDATE_POLICY,
    BTStore,
    evaluate_risk_factor_returns,
    fit_risk_factor_returns,
    plan_risk_factor_update,
)
from bagelquant_bt.risk import risk_interval_returns


def inputs(count=12):
    calendar = [date(2024, 1, 1) + timedelta(days=index) for index in range(count)]
    rng = np.random.default_rng(5)
    exposures, prices = [], []
    for index, day in enumerate(calendar):
        for asset in range(20):
            exposures.append(
                {
                    "time": day,
                    "asset_id": str(asset),
                    "market_cap": float(asset + 1),
                    "industry": "a" if asset < 7 else "b",
                    "size": float(rng.normal()),
                }
            )
            if not (asset == 0 and index in (4, 5, 6)):
                prices.append(
                    {
                        "time": day,
                        "asset_id": str(asset),
                        "price": float(100 + asset + index),
                    }
                )
    return calendar, pl.DataFrame(exposures), pl.DataFrame(prices)


def run(exposures, prices, plan, **kwargs):
    return evaluate_risk_factor_returns(
        exposures,
        prices,
        plan,
        styles=["size"],
        minimum_assets=2,
        observations_per_parameter=1,
        **kwargs,
    )


def test_cold_fit_matches_old_daily_oracle_and_exact_maturity():
    days, exposures, prices = inputs()
    plan = plan_risk_factor_update(days, start=days[0], through=days[-1])
    actual = run(exposures, prices, plan)
    forward = risk_interval_returns(prices, pl.DataFrame({"time": days}))
    rows, diagnostics = [], []
    for index, decision in enumerate(days[:-2]):
        cross = exposures.filter(pl.col("time") == decision).join(
            forward.filter(pl.col("time") == days[index + 1]).select(
                "asset_id", "forward_return"
            ),
            on="asset_id",
            how="left",
        )
        fit = fit_risk_factor_returns(
            cross, styles=["size"], minimum_assets=2, observations_per_parameter=1
        )
        rows.append(
            fit.factor_returns.with_columns(
                pl.lit(days[index + 1]).alias("time"),
                pl.lit(decision).alias("decision_date"),
                pl.lit(days[index + 2]).alias("available_date"),
                pl.lit(True).alias("active"),
                pl.lit(fit.reason).cast(pl.String).alias("reason"),
            )
        )
        diagnostics.append(
            {
                "time": days[index + 1],
                "decision_date": decision,
                "available_date": days[index + 2],
                **{
                    key: value
                    for key, value in asdict(fit).items()
                    if key != "factor_returns"
                },
            }
        )
    assert_frame_equal(
        actual.factor_returns, pl.concat(rows).sort("time", "factor"), check_exact=True
    )
    assert_frame_equal(
        actual.diagnostics,
        pl.DataFrame(diagnostics, schema=actual.diagnostics.schema),
        check_exact=True,
    )
    assert actual.fit_count == len(days) - 2
    assert actual.state["pending_decisions"] == [day.isoformat() for day in days[-2:]]
    assert actual.state["policy"] == RISK_UPDATE_POLICY
    assert actual.state["carry_boundary"] == days[-2].isoformat()


def test_bounded_append_suspended_carry_preserves_history_and_noop(
    tmp_path, monkeypatch
):
    days, exposures, prices = inputs()
    first_plan = plan_risk_factor_update(days[:7], start=days[0], through=days[6])
    first = run(
        exposures.filter(pl.col("time") <= days[6]),
        prices.filter(pl.col("time") <= days[6]),
        first_plan,
    )
    store = BTStore(tmp_path / "bt.sqlite", tmp_path / "artifacts")
    store.initialize()
    saved = store.publish_shared(
        "model",
        "risk",
        "first",
        {
            "returns": first.factor_returns,
            "diagnostics": first.diagnostics,
            "price_carry": first.price_carry,
        },
        metadata={"state": first.state},
    )
    files_before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
        for path in (tmp_path / "artifacts").rglob("*.parquet")
    }
    append = plan_risk_factor_update(
        days, start=days[0], through=days[-1], completed_through=days[6]
    )
    assert append.schedule["decision_date"].min() == days[5]
    observed = []
    from bagelquant_bt import risk_update

    original = risk_update.fit_risk_factor_returns

    def instrumented(frame, **kwargs):
        observed.append(frame)
        return original(frame, **kwargs)

    monkeypatch.setattr(risk_update, "fit_risk_factor_returns", instrumented)
    delta_prices = pl.concat(
        [first.price_carry, prices.filter(pl.col("time") > first_plan.carry_boundary)]
    )
    delta = run(
        exposures.filter(
            pl.col("time").is_in(append.schedule["decision_date"].implode())
        ),
        delta_prices,
        append,
    )
    assert len(observed) == len(days) - 7
    current = pl.concat([first.factor_returns, delta.factor_returns]).sort(
        "time", "factor"
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            BTStore,
            "read_reference",
            lambda *_a, **_k: pytest.fail("append read historical numerical rows"),
        )
        new = store.publish_table(delta.factor_returns)
        reference = store.compose_reference(
            [{"reference": saved["files"]["returns"]}, {"reference": new}]
        )
        store.publish_shared(
            "model",
            "risk",
            "second",
            {},
            file_references={"returns": reference},
            metadata={"state": delta.state},
            expected_shared=saved,
        )
    assert_frame_equal(store.read_reference(reference), current, check_exact=True)
    monkeypatch.setattr(risk_update, "fit_risk_factor_returns", original)
    assert_frame_equal(
        current,
        run(
            exposures,
            prices,
            plan_risk_factor_update(days, start=days[0], through=days[-1]),
        ).factor_returns,
        check_exact=True,
    )
    assert all(
        (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes()) == attributes
        for path, attributes in files_before.items()
    )
    noop = plan_risk_factor_update(
        days, start=days[0], through=days[-1], completed_through=days[-1]
    )
    assert noop.complete and list(noop.windows()) == [] and noop.schedule.is_empty()


def test_affected_maturity_windows_and_population_exclusions():
    days, exposures, prices = inputs()
    plan = plan_risk_factor_update(
        days,
        start=days[0],
        through=days[-1],
        completed_through=days[-1],
        affected_available_date=days[8],
    )
    windows = list(plan.windows(sessions=2))
    assert windows[0].schedule["available_date"].to_list() == days[8:10]
    assert pl.concat([window.schedule for window in windows]).equals(plan.schedule)
    population = exposures.select("time", "asset_id")
    population = pl.concat(
        [population, pl.DataFrame({"time": days, "asset_id": ["missing"] * len(days)})]
    )
    result = run(exposures, prices, plan, population=population)
    reference = run(exposures, prices, plan)
    assert_frame_equal(
        result.factor_returns, reference.factor_returns, check_exact=True
    )
    assert result.diagnostics["eligible_count"].to_list() == [21] * 4
    assert result.diagnostics["excluded_count"].to_list() == [1] * 4


def test_cancellation_does_not_return_false_completion():
    days, exposures, prices = inputs()
    plan = plan_risk_factor_update(days, start=days[0], through=days[-1])

    def canceled():
        raise RuntimeError("canceled")

    with pytest.raises(RuntimeError, match="canceled"):
        run(exposures, prices, plan, check_canceled=canceled)
