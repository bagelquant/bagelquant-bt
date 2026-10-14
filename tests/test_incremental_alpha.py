from datetime import date, timedelta

import polars as pl
import pytest
from bagelquant_core import Domain, Node
from bagelquant_core.hashing import hash_dataframe, hash_mapping
from polars.testing import assert_frame_equal

from bagelquant_bt import alpha_input_start, evaluate_alpha
from bagelquant_bt.exceptions import InputValidationError
from bagelquant_bt.horizon import SessionWindow


def inputs(count, *, revision=False):
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(count)]
    frame = pl.DataFrame(
        [
            {
                "time": day,
                "asset_id": asset,
                "value": float((index * 7 + number * 3) % 13),
            }
            for index, day in enumerate(dates)
            for number, asset in enumerate(["a", "b", "c", "d"])
        ]
    )
    if revision:
        frame = frame.with_columns(
            pl.when((pl.col("time") == dates[5]) & (pl.col("asset_id") == "a"))
            .then(50.0)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    labels = frame.select("time", "asset_id").with_columns(
        (
            (pl.col("time").cast(pl.Int32) % 7 - 3) * 0.001
            + pl.col("asset_id").replace_strict(
                {"a": 0.001, "b": -0.002, "c": 0.003, "d": -0.001}
            )
        ).alias("forward_return"),
        (pl.col("time") + timedelta(days=1)).alias("interval_start"),
        (pl.col("time") + timedelta(days=2)).alias("interval_end"),
        (pl.col("time") + timedelta(days=2)).alias("available_date"),
    )
    node = Node.from_domain(
        frame, Domain(calendar=dates, universe=["a", "b", "c", "d"])
    )

    def proof(first, last, information_cutoff):
        return hash_mapping(
            {
                "value": node.interval_identity(first, last),
                "labels": hash_dataframe(
                    labels.filter(
                        (pl.col("time") >= first)
                        & (pl.col("time") <= last)
                        & (pl.col("available_date") <= information_cutoff)
                    )
                ),
            }
        )

    return node, labels, pl.DataFrame({"time": dates}), proof


SETTINGS = {
    "windows": (
        SessionWindow("cumulative", "1", 1, 1),
        SessionWindow("cumulative", "3", 1, 3),
        SessionWindow("bucket", "2_5", 2, 5),
    ),
    "persistence_lags": (1, 3, 5),
    "lead_lags": (-3, 0, 2),
    "alpha_return_lags": (0, 2),
    "quantiles": 2,
    "rolling_observations": 5,
}


@pytest.mark.parametrize(
    "components",
    [("book_tail",), ("turnover",), ("quantiles",), ("lead_lag",), ("alpha_return",)],
)
def test_first_mature_path_restores_original_unprocessed_decisions(components):
    alpha, labels, calendar, _proof = inputs(35)
    # An owner token binds unchanged original labels at each information cutoff.
    def proof(*args):
        return "unchanged-input-prefix"
    cutoff = date(2024, 2, 5)
    labels = labels.with_columns(
        pl.max_horizontal("interval_end", pl.lit(cutoff)).alias("available_date")
    )
    previous = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=date(2024, 2, 4),
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    assert all(value is None for value in previous.state["paths"].values())
    alpha, labels, calendar, _ = inputs(36)
    labels = labels.with_columns(
        pl.max_horizontal("interval_end", pl.lit(cutoff)).alias("available_date")
    )
    assert (
        alpha_input_start(previous, calendar=calendar, components=components)
        == calendar["time"].min()
    )
    updated = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=cutoff,
        components=components,
        previous=previous,
        input_proof=proof,
        **SETTINGS,
    )
    expected = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=cutoff,
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    for name, table in expected.tables.items():
        assert_frame_equal(
            updated.tables[name], table, check_exact=False, rel_tol=1e-12, abs_tol=1e-12
        )


@pytest.mark.parametrize(
    "components", [None, ("persistence",), ("book_tail", "turnover")]
)
def test_bounded_input_consumption_matches_complete_inputs(components):
    first, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        first,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    alpha, labels, calendar, proof = inputs(36)
    first = alpha_input_start(
        previous,
        calendar=calendar,
        components=components,
        **{
            name: SETTINGS[name]
            for name in (
                "windows",
                "persistence_lags",
                "lead_lags",
                "alpha_return_lags",
            )
        },
    )
    assert first > calendar["time"].min()
    selected = alpha.collect(dense=False).filter(pl.col("time") >= first)
    tail = Node.from_domain(selected, alpha.domain)
    updated = evaluate_alpha(
        tail,
        labels.filter(pl.col("time") >= first),
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        previous=previous,
        **SETTINGS,
    )
    reference = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    assert updated.tables.keys() == reference.tables.keys()
    for name, expected in reference.tables.items():
        keys = [
            column
            for column in (
                "time",
                "evaluation_date",
                "target_date",
                "window_kind",
                "window_id",
                "lag",
                "path_kind",
                "quantile",
                "horizon_sessions",
            )
            if column in expected.columns
        ]
        assert_frame_equal(
            updated.tables[name].select(expected.columns).sort(keys),
            expected.sort(keys),
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


@pytest.mark.parametrize(
    "before,after",
    [
        (("persistence",), None),
        (("book_tail",), None),
        (("horizons",), ("horizons", "rolling_ic")),
        (("turnover",), ("turnover", "book_tail")),
        (("rolling_ic",), ("rolling_ic", "horizons")),
    ],
)
@pytest.mark.parametrize("append", [False, True])
def test_component_upgrade_preserves_completed_families(
    before, after, append, monkeypatch
):
    alpha, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=before,
        input_proof=proof,
        **SETTINGS,
    )
    if append:
        alpha, labels, calendar, proof = inputs(36)
    from bagelquant_bt import evaluation

    original = evaluation.simulate_weight_returns
    starts = []

    def observed(weights, returns, **kwargs):
        checkpoint = kwargs.get("checkpoint")
        if checkpoint is not None:
            starts.append((checkpoint.time, returns["time"].min()))
        return original(weights, returns, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "simulate_weight_returns", observed)
        updated = evaluate_alpha(
            alpha,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            components=after,
            previous=previous,
            input_proof=proof,
            **SETTINGS,
        )
    reference = evaluate_alpha(
        alpha,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=after,
        input_proof=proof,
        **SETTINGS,
    )
    assert updated.tables.keys() == reference.tables.keys()
    for name, expected in reference.tables.items():
        keys = [
            column
            for column in (
                "time",
                "evaluation_date",
                "target_date",
                "window_kind",
                "window_id",
                "lag",
                "path_kind",
                "quantile",
                "horizon_sessions",
            )
            if column in expected.columns
        ]
        assert_frame_equal(
            updated.tables[name].select(expected.columns).sort(keys),
            expected.sort(keys),
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    assert all(first is None or first > completed for completed, first in starts)
    assert updated.state["components"] == reference.state["components"]


@pytest.mark.parametrize(
    "components",
    [
        ("horizons", "rolling_ic", "persistence"),
        ("book_tail", "turnover", "quantiles", "lead_lag", "alpha_return"),
        None,
    ],
)
def test_append_reuses_mature_statistics_and_continues_weight_state(
    components, monkeypatch
):
    first, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        first,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    appended, labels, calendar, proof = inputs(36)
    from bagelquant_bt import evaluation

    original_ic = evaluation.window_information_coefficients
    ic_dates = []

    def observed(factor, returns):
        ic_dates.append(factor["evaluation_date"].unique().sort().to_list())
        return original_ic(factor, returns)

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "window_information_coefficients", observed)
        updated = evaluate_alpha(
            appended,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            components=components,
            input_proof=proof,
            previous=previous,
            **SETTINGS,
        )
    reference = evaluate_alpha(
        appended,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        **SETTINGS,
    )
    assert updated.tables.keys() == reference.tables.keys()
    for name, table in reference.tables.items():
        actual = updated.tables[name]
        keys = [
            column
            for column in (
                "time",
                "evaluation_date",
                "target_date",
                "window_kind",
                "window_id",
                "lag",
                "path_kind",
                "quantile",
                "horizon_sessions",
            )
            if column in table.columns
        ]
        assert_frame_equal(
            actual.select(table.columns).sort(keys),
            table.sort(keys),
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    if ic_dates:
        assert all(min(days) >= date(2024, 1, 30) for days in ic_dates if days)
    assert updated.execution["path_sessions"] <= len(updated.state["paths"])
    assert updated.execution["continued_paths"] == sum(
        item is not None for item in previous.state["paths"].values()
    )


def test_changed_prefix_and_missing_proof_are_actionable_before_daily_work(monkeypatch):
    first, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        first,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        **SETTINGS,
    )
    revised, labels, calendar, proof = inputs(36, revision=True)
    monkeypatch.setattr(
        "bagelquant_bt.evaluation.simulate_weight_returns",
        lambda *_a, **_k: pytest.fail("replayed historical paths"),
    )
    with pytest.raises(InputValidationError, match="input prefix changed"):
        evaluate_alpha(
            revised,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            input_proof=proof,
            previous=previous,
            **SETTINGS,
        )


def test_delayed_maturity_resumes_from_actual_path_boundary():
    components = ("book_tail", "turnover", "quantiles", "lead_lag", "alpha_return")

    def build(count):
        node, labels, calendar, _proof = inputs(count)
        labels = labels.with_columns(
            pl.when(pl.col("time") == date(2024, 1, 5))
            .then(pl.lit(date(2024, 2, 5)))
            .otherwise(pl.col("available_date"))
            .alias("available_date")
        )

        def proof(first, last, information_cutoff):
            return hash_mapping(
                {
                    "value": node.interval_identity(first, last),
                    "labels": hash_dataframe(
                        labels.filter(
                            (pl.col("time") >= first)
                            & (pl.col("time") <= last)
                            & (pl.col("available_date") <= information_cutoff)
                        )
                    ),
                }
            )

        return evaluate_alpha(
            node,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            components=components,
            input_proof=proof,
            **SETTINGS,
        ), (node, labels, calendar, proof)

    previous, _ = build(35)
    reference, (node, labels, calendar, proof) = build(36)
    assert previous.state["paths"]["book"]["time"] == "2024-01-04"
    updated = evaluate_alpha(
        node,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        components=components,
        input_proof=proof,
        previous=previous,
        **SETTINGS,
    )
    for name, table in reference.tables.items():
        keys = [
            column
            for column in ("time", "lag", "path_kind", "quantile")
            if column in table.columns
        ]
        assert_frame_equal(
            updated.tables[name].sort(keys),
            table.sort(keys),
            check_exact=False,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_noop_backward_cutoff_and_missing_path_table_do_no_daily_work(monkeypatch):
    from dataclasses import replace

    node, labels, calendar, proof = inputs(35)
    previous = evaluate_alpha(
        node,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        **SETTINGS,
    )
    monkeypatch.setattr(
        "bagelquant_bt.evaluation.simulate_weight_returns",
        lambda *_a, **_k: pytest.fail("ran daily work"),
    )
    current = evaluate_alpha(
        node,
        labels,
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        previous=previous,
        **SETTINGS,
    )
    assert current.execution["outcome"] == "up_to_date"
    with pytest.raises(InputValidationError, match="nonregressing"):
        evaluate_alpha(
            node,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max() - timedelta(days=1),
            input_proof=proof,
            previous=previous,
            **SETTINGS,
        )
    damaged = replace(
        previous,
        tables={
            name: table
            for name, table in previous.tables.items()
            if name != "state_book_returns"
        },
    )
    with pytest.raises(InputValidationError, match="tables are incomplete"):
        evaluate_alpha(
            node,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            input_proof=proof,
            previous=damaged,
            **SETTINGS,
        )
    with pytest.raises(InputValidationError, match="explicit maintenance"):
        evaluate_alpha(
            node,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            previous=previous,
            **SETTINGS,
        )
