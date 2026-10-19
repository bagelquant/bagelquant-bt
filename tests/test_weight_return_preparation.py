"""Read-only return preparation never shares account state or changes axes."""

from datetime import date

import polars as pl
import pytest
from bagelquant_core.resources import ResourceLimits
from polars.testing import assert_frame_equal

from bagelquant_bt import BTExecutionOptions, evaluate_alpha
from bagelquant_bt.portfolio_mechanics import (
    _WeightReturnPreparation,
    normalize_forward_returns,
    simulate_weight_returns,
)
from test_incremental_alpha import SETTINGS, inputs


def assert_paths_equal(actual, expected):
    left, right = actual
    old, prior = expected
    assert left.keys() == old.keys()
    for name in left:
        assert_frame_equal(left[name], old[name], check_exact=True)
    if right is None or prior is None:
        assert right is prior
    else:
        for name in (
            "time",
            "interval_end",
            "available_date",
            "gross_value",
            "net_value",
        ):
            assert getattr(right, name) == getattr(prior, name)
        assert_frame_equal(right.weights, prior.weights, check_exact=True)


def test_shared_panels_preserve_windows_nulls_state_and_later_asset_axes():
    alpha, raw, calendar, _ = inputs(12)
    raw = raw.filter(
        ~((pl.col("asset_id") == "d") & (pl.col("time") < date(2024, 1, 8)))
    )
    raw = raw.with_columns(
        pl.when((pl.col("asset_id") == "b") & (pl.col("time") == date(2024, 1, 3)))
        .then(None)
        .otherwise(pl.col("forward_return"))
        .alias("forward_return")
    )
    labels = normalize_forward_returns(
        raw, calendar=calendar, available_date=date(2024, 1, 20)
    )
    weights = alpha.collect(dense=False).with_columns(
        (pl.col("value") / 20).alias("value")
    )
    prepared = _WeightReturnPreparation(labels, 1024 * 1024)
    for through in (date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 10)):
        window = prepared.window(through=through)
        selected = weights.filter(pl.col("time") <= through)
        if through.day == 5:
            selected = selected.filter(pl.col("asset_id") != "d")
        options = {"decisions": None, "cost_rate": 0.0005, "include_holdings": True}
        expected = simulate_weight_returns(selected, window.frame, **options)
        assert_paths_equal(
            simulate_weight_returns(
                selected, window.frame, _prepared=window, **options
            ),
            expected,
        )
        # Repeated callers may change costs without borrowing mutable holdings.
        changed = {**options, "cost_rate": 0.001}
        assert_paths_equal(
            simulate_weight_returns(
                selected, window.frame, _prepared=window, **changed
            ),
            simulate_weight_returns(selected, window.frame, **changed),
        )
    assert prepared.hits >= 3
    assert prepared.bytes <= prepared.maximum_bytes
    for panels, _ in prepared._panels.values():
        for panel in panels:
            assert not panel.indices.flags.writeable
            assert not panel.returns.flags.writeable
            assert not panel.observed.flags.writeable
    prepared.clear()
    assert prepared.bytes == 0 and not prepared._panels


def test_budget_and_frame_binding_reject_foreign_or_unadmitted_preparation():
    _, raw, calendar, _ = inputs(5)
    labels = normalize_forward_returns(
        raw, calendar=calendar, available_date=date(2024, 1, 20)
    )
    prepared = _WeightReturnPreparation(labels, 0)
    window = prepared.window()
    list(prepared.panels(window.frame, ["a", "b", "c", "d"], None))
    assert prepared.bytes == 0 and not prepared._panels
    empty = pl.DataFrame(
        schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64}
    )
    with pytest.raises(ValueError, match="different selected"):
        simulate_weight_returns(
            empty, window.frame.clone(), decisions=None, cost_rate=0.0, _prepared=window
        )
    bounded = _WeightReturnPreparation(labels, 1024 * 1024, after=date(2024, 1, 3))
    with pytest.raises(ValueError, match="extend"):
        bounded.window(after=date(2024, 1, 2))


@pytest.mark.parametrize("budget", [0, 128, 4096])
def test_fallback_keeps_no_owned_source_frame_or_asset_set(budget, monkeypatch):
    _, raw, calendar, _ = inputs(20)
    labels = normalize_forward_returns(
        raw, calendar=calendar, available_date=date(2024, 2, 20)
    )
    calls = []
    original = pl.DataFrame.filter

    def observe(frame, *args, **kwargs):
        calls.append(frame.height)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "filter", observe)
    prepared = _WeightReturnPreparation(labels, budget, after=date(2024, 1, 3))
    assert calls == []
    window = prepared.window(after=date(2024, 1, 3))
    calls.clear()
    panels = list(prepared.panels(window.frame, ["a", "b", "c", "d"], None))
    assert len(panels) == 17 and calls == []
    assert prepared.bytes == 0 and prepared.peak_bytes == 0
    assert not prepared._panels
    assert not hasattr(prepared, "source") and not hasattr(prepared, "source_assets")
    prepared.clear()
    assert prepared.labels is None


def test_distinct_axes_evict_and_cancellation_does_not_retain_partial_panels():
    _, raw, calendar, _ = inputs(12)
    labels = normalize_forward_returns(
        raw, calendar=calendar, available_date=date(2024, 2, 20)
    )
    axes = ["a", "b", "c", "d", "e"]
    prepared = _WeightReturnPreparation(
        labels, _WeightReturnPreparation._charge(labels, axes)
    )
    window = prepared.window()
    list(prepared.panels(window.frame, axes[:-1], None))
    list(prepared.panels(window.frame, axes, None))
    assert len(prepared._panels) == 1
    assert prepared.bytes <= prepared.maximum_bytes
    prepared.clear()
    prepared = _WeightReturnPreparation(labels, 1024 * 1024)

    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("canceled")

    with pytest.raises(RuntimeError, match="canceled"):
        prepared.panels(prepared.window().frame, axes, cancel)
    assert prepared.bytes == 0 and not prepared._panels
    assert len(list(prepared.panels(prepared.window().frame, axes, None))) == 12


def test_complete_alpha_cache_and_incremental_preparation_preserve_exact_outputs():
    previous = None
    for sessions in (35, 37):
        alpha, labels, calendar, proof = inputs(sessions)
        results = []
        for cache in (0, 16):
            options = BTExecutionOptions(
                limits=ResourceLimits(
                    total_threads=1, memory_target_mib=512, cache_mib=cache
                )
            )
            results.append(
                evaluate_alpha(
                    alpha,
                    labels,
                    calendar=calendar,
                    available_date=calendar["time"].max(),
                    input_proof=proof,
                    previous=previous,
                    options=options,
                    **SETTINGS,
                )
            )
        for name in results[0].tables:
            assert_frame_equal(
                results[0].tables[name], results[1].tables[name], check_exact=True
            )
        assert results[0].state == results[1].state
        assert (
            results[0].execution["path_sessions"]
            == results[1].execution["path_sessions"]
        )
        assert results[1].execution["return_preparation_hits"] > 0
        assert (
            results[1].execution["return_prepared_rows"]
            < results[0].execution["return_prepared_rows"]
        )
        if previous is not None:
            assert results[1].execution["return_prepared_rows"] <= 4 * (2 + 3 + 2)
        previous = results[1]


def test_alpha_failure_releases_owned_preparation(monkeypatch):
    from bagelquant_bt import evaluation

    alpha, labels, calendar, proof = inputs(12)
    owned = []

    class Observed(_WeightReturnPreparation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            owned.append(self)

    monkeypatch.setattr(evaluation, "_WeightReturnPreparation", Observed)

    def fail(*_args, **_kwargs):
        raise RuntimeError("interrupted causal path")

    monkeypatch.setattr(evaluation, "simulate_weight_returns", fail)
    with pytest.raises(RuntimeError, match="interrupted causal path"):
        evaluate_alpha(
            alpha,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            components=("book_tail",),
            input_proof=proof,
            **SETTINGS,
        )
    assert len(owned) == 1 and owned[0].bytes == 0 and not owned[0]._panels
