from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from bagelquant_core import Domain, Node
from polars.testing import assert_frame_equal

from bagelquant_bt.evaluation import (
    evaluate_alpha,
    evaluate_weights,
    return_statistics,
)
from bagelquant_bt.exceptions import InputValidationError
from bagelquant_bt.horizon import SessionWindow
from bagelquant_bt.periods import slice_primitives
from bagelquant_bt.portfolio_mechanics import build_alpha_weights


def _node(values, *, value_type="numeric", calendar=None):
    rows = [
        (day, asset, value)
        for day, cross in values.items()
        for asset, value in cross.items()
    ]
    frame = pl.DataFrame(
        rows,
        schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64},
        orient="row",
    )
    return Node.from_domain(
        frame,
        Domain(
            calendar=sorted(values) if calendar is None else calendar,
            universe=frame["asset_id"].unique().sort(),
        ),
        value_type=value_type,
    )


def _labels(values, *, immature=()):
    rows = [
        {
            "time": day,
            "asset_id": asset,
            "forward_return": value,
            "interval_start": day + timedelta(days=1),
            "interval_end": day + timedelta(days=2),
            "available_date": day + timedelta(days=2 if day not in immature else 20),
        }
        for day, cross in values.items()
        for asset, value in cross.items()
    ]
    return pl.DataFrame(
        rows,
        schema={
            "time": pl.Date,
            "asset_id": pl.String,
            "forward_return": pl.Float64,
            "interval_start": pl.Date,
            "interval_end": pl.Date,
            "available_date": pl.Date,
        },
    )


DATES = [date(2024, 1, 1) + timedelta(days=index) for index in range(8)]


@pytest.mark.parametrize("value_type", ["numeric", "prediction"])
def test_alpha_uses_caller_start_label_without_execution_shift(value_type):
    day = DATES[0]
    alpha = _node({day: {"a": 1.0, "b": 2.0, "c": 3.0}}, value_type=value_type)
    returns = _labels({day: {"a": 0.1, "b": 0.2, "c": 0.3}})
    result = evaluate_alpha(
        alpha,
        returns,
        components=("horizons",),
        quantiles=2,
        windows=(SessionWindow("cumulative", "1D", 1, 1),),
        persistence_lags=(1,),
    )
    ic = result.tables["horizon_ic"]
    assert ic["evaluation_date"].to_list() == [day]
    assert ic["target_end_date"].to_list() == [day + timedelta(days=2)]
    assert ic["pearson_ic"][0] == pytest.approx(1)
    assert ic["spearman_ic"][0] == pytest.approx(1)
    assert result.tables["horizon_ic_summary"]["icir"].null_count() == 2
    assert result.tables["horizon_ic_summary"]["annualized_icir"].null_count() == 2


def test_book_and_spread_are_gross_one_net_zero_and_ties_are_stable():
    day = DATES[0]
    alpha = _node({day: {"b": 1.0, "a": 1.0, "c": 2.0, "d": 4.0}})
    book = build_alpha_weights(alpha).collect()
    assert book["value"].to_list() == pytest.approx([-0.25, -0.25, 0.125, 0.375])
    spread = build_alpha_weights(alpha, method="spread", quantiles=2).collect()
    assert spread["value"].to_list() == pytest.approx([-0.25, -0.25, 0.25, 0.25])
    for panel in (book, spread):
        assert panel["value"].sum() == pytest.approx(0)
        assert panel["value"].abs().sum() == pytest.approx(1)


def test_alpha_builder_uses_latest_whole_snapshot_on_fixed_cadence():
    alpha = _node(
        {
            DATES[0]: {"a": 1.0, "b": 2.0},
            DATES[1]: {"a": 4.0, "b": 1.0},
            DATES[3]: {"a": 1.0, "b": 4.0},
        },
        calendar=DATES[:5],
    )
    built = build_alpha_weights(
        alpha, every=2, anchor=DATES[0], calendar=pl.DataFrame({"time": DATES[:5]})
    ).collect(dense=False)
    assert built["time"].unique().sort().to_list() == [DATES[0], DATES[2], DATES[4]]
    assert built.filter(pl.col("time") == DATES[2])["value"].to_list() == [0.5, -0.5]
    with pytest.raises(InputValidationError, match="calendar and anchor"):
        build_alpha_weights(alpha, every=2)


def test_alpha_horizon_compounds_supplied_returns_and_bucket_offsets():
    alpha = _node({DATES[0]: {"a": 1.0, "b": 2.0}}, calendar=DATES[:3])
    returns = _labels(
        {
            DATES[0]: {"a": 0.1, "b": 0.0},
            DATES[1]: {"a": 0.2, "b": 0.1},
            DATES[2]: {"a": 0.3, "b": 0.2},
        }
    )
    result = evaluate_alpha(
        alpha,
        returns,
        components=("horizons",),
        quantiles=2,
        persistence_lags=(1,),
        windows=(
            SessionWindow("cumulative", "2D", 1, 2),
            SessionWindow("bucket", "2-3D", 2, 3),
        ),
    )
    book = result.tables["horizon_book_returns"].sort("window_id")
    assert book.filter(pl.col("window_id") == "2D")["book_return"][0] == pytest.approx(
        -0.11
    )
    assert book.filter(pl.col("window_id") == "2-3D")["book_return"][
        0
    ] == pytest.approx(-0.12)


def test_mature_missing_member_keeps_original_book_weight_but_immature_is_excluded():
    values = {day: {"a": 1.0, "b": 3.0} for day in DATES[:2]}
    alpha = _node(values)
    labels = _labels(
        {DATES[0]: {"a": None, "b": 0.2}, DATES[1]: {"a": 0.3, "b": 0.4}},
        immature=(DATES[1],),
    )
    result = evaluate_alpha(
        alpha,
        labels,
        available_date=DATES[2],
        quantiles=2,
        components=("horizons", "book_tail"),
        persistence_lags=(1,),
        windows=(SessionWindow("cumulative", "1D", 1, 1),),
    )
    book = result.tables["horizon_book_returns"]
    assert book["evaluation_date"].to_list() == [DATES[0]]
    assert book["book_return"][0] == pytest.approx(0.1)
    assert book["coverage_ratio"][0] == 0.5
    assert result.tables["daily_book_returns"]["time"].to_list() == [DATES[0]]
    assert result.tables["daily_book_returns"]["gross_return"][0] == pytest.approx(0.1)
    coverage = result.tables["horizon_coverage"].sort("evaluation_date")
    assert coverage["is_mature"].to_list() == [True, False]


def test_alpha_streaming_window_has_strict_missing_daily_member_coverage():
    alpha = _node({DATES[0]: {"a": 1.0, "b": 2.0}}, calendar=DATES[:3])
    labels = _labels(
        {
            DATES[0]: {"a": 0.1, "b": 0.1},
            DATES[1]: {"b": 0.2},
            DATES[2]: {"a": 0.3, "b": 0.3},
        }
    )
    result = evaluate_alpha(
        alpha,
        labels,
        components=("horizons",),
        quantiles=2,
        persistence_lags=(1,),
        windows=(SessionWindow("cumulative", "3D", 1, 3),),
    )
    book = result.tables["horizon_book_returns"]
    assert book["expected_count"][0] == 2
    assert book["observed_count"][0] == 1
    assert book["book_return"][0] == pytest.approx(0.5 * (1.1 * 1.2 * 1.3 - 1))


def test_weight_holdings_drift_and_cost_uses_net_pretrade_full_l1():
    weights = _node(
        {DATES[0]: {"a": 0.5, "b": 0.5}, DATES[2]: {"a": 0.5, "b": 0.5}},
        value_type="weights",
        calendar=DATES[:3],
    )
    labels = _labels(
        {
            DATES[0]: {"a": 0.2, "b": 0.0},
            DATES[1]: {"a": 0.1, "b": 0.0},
            DATES[2]: {"a": 0.0, "b": 0.0},
        }
    )
    result = evaluate_weights(
        weights, labels, components=("returns",), cost_rate=0.0005
    )
    returns = result.tables["returns"]
    assert returns["gross_return"][0] == pytest.approx(0.1)
    assert returns["net_return"][0] == pytest.approx(0.0995)
    assert returns["net_pre_cost_return"][1] == pytest.approx(0.6 / 1.0995 * 0.1)
    assert returns["gross_return"][1] == pytest.approx(0.6 / 1.1 * 0.1)
    gross_after = np.array([0.66, 0.5]) / 1.16
    net_after = np.array([0.66, 0.5]) / 1.1595
    expected_turnover = np.abs(np.array([0.5, 0.5]) - net_after).sum()
    assert result.tables["turnover"]["turnover"][2] == pytest.approx(expected_turnover)
    assert returns["net_return"][2] == pytest.approx(-0.0005 * expected_turnover)
    held = result.tables["weights"].filter(pl.col("time") == DATES[1])
    assert held["gross_weight"].to_list() == pytest.approx(gross_after)
    assert held["net_weight"].to_list() == pytest.approx(net_after)


def test_weight_checkpoint_suffix_matches_full_run_with_independent_drift():
    values = {DATES[0]: {"a": 0.5, "b": 0.5}, DATES[2]: {"a": 0.2, "b": 0.8}}
    weights = _node(values, value_type="weights", calendar=DATES[:4])
    labels = _labels({day: {"a": 0.05, "b": -0.01} for day in DATES[:4]})
    full = evaluate_weights(weights, labels, components=("returns",))
    first = evaluate_weights(
        weights, labels.filter(pl.col("time") <= DATES[1]), components=("returns",)
    )
    suffix = evaluate_weights(
        weights, labels, components=("returns",), checkpoint=first.checkpoint
    )
    assert_frame_equal(
        pl.concat([first.tables["returns"], suffix.tables["returns"]]),
        full.tables["returns"],
    )
    assert_frame_equal(full.checkpoint.weights, suffix.checkpoint.weights)
    assert suffix.checkpoint.net_value == pytest.approx(full.checkpoint.net_value)


def test_weight_mature_missing_asset_returns_zero_without_reselecting():
    weights = _node(
        {DATES[0]: {"a": 0.5, "b": 0.5}}, value_type="weights", calendar=DATES[:2]
    )
    labels = _labels({DATES[0]: {"a": 0.1, "b": None}, DATES[1]: {"a": 0.0, "b": 0.2}})
    result = evaluate_weights(weights, labels, cost_rate=0, components=("returns",))
    assert result.tables["returns"]["gross_return"].to_list() == pytest.approx(
        [0.05, 0.5 / 1.05 * 0.2]
    )
    assert result.tables["coverage"]["coverage_ratio"].to_list() == [0.5, 1.0]


def test_weight_lag_common_sample_shifts_saved_snapshots_explicitly():
    weights = _node({day: {"a": 1.0} for day in DATES[:4]}, value_type="weights")
    labels = _labels({day: {"a": 0.1 * index} for index, day in enumerate(DATES[:4])})
    result = evaluate_weights(weights, labels, lags=(0, 1), cost_rate=0)
    lags = result.tables["lag_returns"]
    assert lags.filter(pl.col("lag") == 0)["time"].to_list() == DATES[1:4]
    assert lags.filter(pl.col("lag") == 1)["time"].to_list() == DATES[1:4]


def test_alpha_all_components_share_existing_aggregate_schemas():
    values = {
        day: {f"a{index}": float(index + day.day % 3) for index in range(10)}
        for day in DATES
    }
    labels = _labels(
        {
            day: {f"a{index}": (index - 4) * 0.001 for index in range(10)}
            for day in DATES
        }
    )
    result = evaluate_alpha(
        _node(values),
        labels,
        windows=(SessionWindow("cumulative", "1D", 1, 1),),
        persistence_lags=(1, 2),
        lead_lags=(-1, 0, 1),
        alpha_return_lags=(0, 1),
        rolling_observations=2,
    )
    expected = {
        "horizon_ic",
        "horizon_ic_summary",
        "horizon_book_returns",
        "horizon_tail_returns",
        "horizon_quantile_forward_returns",
        "horizon_quantile_structure",
        "horizon_factor_returns",
        "horizon_statistical_inference",
        "horizon_signal_persistence",
        "daily_signal_autocorrelation",
        "daily_book_returns",
        "daily_tail_returns",
        "daily_book_turnover",
        "daily_book_lead_lag_returns",
        "daily_alpha_return_lag_returns",
        "daily_quantile_returns",
        "daily_rolling_ic",
    }
    assert expected <= result.tables.keys()
    assert result.tables["daily_quantile_returns"]["quantile"].n_unique() == 10
    assert result.tables["horizon_statistical_inference"].height > 0
    assert result.tables["daily_book_lead_lag_returns"].height == 18


@pytest.mark.parametrize("column", ["interval_start", "interval_end", "available_date"])
def test_explicit_label_metadata_is_required(column):
    alpha = _node({DATES[0]: {"a": 1.0, "b": 2.0}})
    labels = _labels({DATES[0]: {"a": 0.1, "b": 0.2}}).drop(column)
    with pytest.raises(InputValidationError, match="interval_end and available_date"):
        evaluate_alpha(
            alpha, labels, components=("persistence",), persistence_lags=(1,)
        )


def test_constant_alpha_is_unavailable_and_wrong_node_type_fails():
    alpha = _node({DATES[0]: {"a": 1.0, "b": 1.0}})
    assert build_alpha_weights(alpha).collect()["value"].null_count() == 2
    labels = _labels({DATES[0]: {"a": 0.1, "b": 0.2}})
    result = evaluate_alpha(
        alpha,
        labels,
        components=("horizons",),
        quantiles=2,
        persistence_lags=(1,),
        windows=(SessionWindow("cumulative", "1D", 1, 1),),
    )
    assert result.tables["horizon_ic"]["pearson_ic"][0] is None
    with pytest.raises(TypeError, match="weights must be"):
        evaluate_weights(alpha, labels)


def test_return_statistics_includes_first_loss_drawdown_and_null_ratios():
    frame = pl.DataFrame({"gross_return": [-0.1, 0.1], "net_return": [0.0, 0.0]})
    metrics = return_statistics(frame, annualization=2)
    assert metrics["gross_total_return"] == pytest.approx(-0.01)
    assert metrics["gross_max_drawdown"] == pytest.approx(-0.1)
    assert metrics["net_sharpe"] is None
    assert metrics["net_calmar"] is None
    assert metrics["net_sample_size"] == 2


def test_alignment_key_and_economic_interval_are_distinct():
    alpha = _node({DATES[0]: {"a": 1.0, "b": 2.0}})
    labels = _labels({DATES[0]: {"a": 0.1, "b": 0.2}})
    result = evaluate_alpha(
        alpha,
        labels,
        components=("horizons",),
        quantiles=2,
        windows=(SessionWindow("cumulative", "1D", 1, 1),),
        persistence_lags=(1,),
    )
    ic = result.tables["horizon_ic"]
    assert ic["evaluation_date"].to_list() == [DATES[0]]
    assert ic["target_start_date"].to_list() == [DATES[1]]
    assert ic["target_end_date"].to_list() == [DATES[2]]


def test_overlapping_economic_intervals_fail_without_a_hidden_date_shift():
    weights = _node({day: {"a": 1.0} for day in DATES[:2]}, value_type="weights")
    labels = _labels({day: {"a": 0.1} for day in DATES[:2]}).with_columns(
        (pl.col("interval_end") + pl.duration(days=1)).alias("interval_end"),
        (pl.col("available_date") + pl.duration(days=1)).alias("available_date"),
    )
    with pytest.raises(InputValidationError, match="must not overlap"):
        evaluate_weights(weights, labels, components=("returns",))


def test_gap_between_economic_intervals_does_not_form_a_horizon_label():
    alpha = _node({DATES[0]: {"a": 1.0, "b": 2.0}}, calendar=DATES[:2])
    labels = _labels({day: {"a": 0.1, "b": 0.2} for day in DATES[:2]}).with_columns(
        pl.when(pl.col("time") == DATES[1])
        .then(pl.col("interval_start") + pl.duration(days=1))
        .otherwise(pl.col("interval_start"))
        .alias("interval_start"),
        pl.when(pl.col("time") == DATES[1])
        .then(pl.col("interval_end") + pl.duration(days=1))
        .otherwise(pl.col("interval_end"))
        .alias("interval_end"),
        pl.when(pl.col("time") == DATES[1])
        .then(pl.col("available_date") + pl.duration(days=1))
        .otherwise(pl.col("available_date"))
        .alias("available_date"),
    )
    result = evaluate_alpha(
        alpha,
        labels,
        components=("horizons",),
        quantiles=2,
        windows=(SessionWindow("cumulative", "2D", 1, 2),),
        persistence_lags=(1,),
    )
    assert result.tables["horizon_ic"].is_empty()
    coverage = result.tables["horizon_coverage"]
    assert coverage["is_mature"].to_list() == [True]
    assert coverage["is_complete"].to_list() == [False]
    assert coverage["unavailable_reason"][0] == "economic intervals are not contiguous"


def test_checkpoint_rejects_a_suffix_that_overlaps_the_prior_endpoint():
    weights = _node({day: {"a": 1.0} for day in DATES[:2]}, value_type="weights")
    labels = _labels({day: {"a": 0.1} for day in DATES[:2]})
    initial = evaluate_weights(
        weights,
        labels.filter(pl.col("time") == DATES[0]),
        components=("returns",),
    )
    suffix = labels.filter(pl.col("time") == DATES[1]).with_columns(
        pl.lit(DATES[1]).alias("interval_start")
    )
    with pytest.raises(InputValidationError, match="contiguous economic intervals"):
        evaluate_weights(
            weights, suffix, components=("returns",), checkpoint=initial.checkpoint
        )


def test_icir_exposes_raw_and_annualized_ratios_separately():
    alpha = _node({day: {"a": 1.0, "b": 2.0} for day in DATES[:3]})
    labels = _labels(
        {
            DATES[0]: {"a": 0.0, "b": 0.1},
            DATES[1]: {"a": 0.1, "b": 0.0},
            DATES[2]: {"a": 0.0, "b": 0.1},
        }
    )
    result = evaluate_alpha(
        alpha,
        labels,
        components=("horizons",),
        quantiles=2,
        windows=(SessionWindow("cumulative", "1D", 1, 1),),
        persistence_lags=(1,),
    )
    summary = result.tables["horizon_ic_summary"]
    expected = np.mean([1, -1, 1]) / np.std([1, -1, 1], ddof=1)
    assert summary["icir"].to_list() == pytest.approx([expected, expected])
    assert summary["annualized_icir"].to_list() == pytest.approx(
        [expected * np.sqrt(252)] * 2
    )


def test_saved_economic_primitives_preserve_delayed_information_availability():
    values = {DATES[0]: {"a": 1.0, "b": 2.0}}
    labels = _labels({DATES[0]: {"a": 0.1, "b": 0.2}}).with_columns(
        pl.lit(DATES[3]).alias("available_date")
    )
    alpha_result = evaluate_alpha(
        _node(values),
        labels,
        available_date=DATES[3],
        components=("horizons", "book_tail", "turnover", "rolling_ic"),
        windows=(SessionWindow("cumulative", "1D", 1, 1),),
        quantiles=2,
        persistence_lags=(1,),
        rolling_observations=1,
    )
    observations = {
        name: frame
        for name, frame in alpha_result.tables.items()
        if name
        not in {
            "daily_signal_autocorrelation",
            "horizon_signal_persistence",
            "horizon_signal_persistence_summary",
        }
    }
    for name, frame in observations.items():
        assert "available_date" in frame.columns, name
        assert (
            frame["available_date"].drop_nulls().to_list() == [DATES[3]] * frame.height
        )
    assert all(
        frame.is_empty()
        for frame in slice_primitives(observations, DATES[0], DATES[2]).values()
    )
    assert slice_primitives(observations, DATES[0], DATES[3])["horizon_ic"].height == 1
    weights_result = evaluate_weights(
        _node(values, value_type="weights"),
        labels,
        available_date=DATES[3],
        components=("returns",),
    )
    economic = {
        name: frame
        for name, frame in weights_result.tables.items()
        if name != "target_weights"
    }
    assert all(
        frame.is_empty()
        for frame in slice_primitives(economic, DATES[0], DATES[2]).values()
    )
    assert weights_result.checkpoint.available_date == DATES[3]
    with pytest.raises(InputValidationError, match="beyond available_date"):
        evaluate_weights(
            _node(values, value_type="weights"),
            labels,
            available_date=DATES[2],
            components=("returns",),
            checkpoint=weights_result.checkpoint,
        )


def test_cash_only_weight_snapshot_keeps_typed_empty_holdings_until_invested():
    weights = _node({DATES[0]: {"a": 0.0}, DATES[1]: {"a": 1.0}}, value_type="weights")
    labels = _labels({DATES[0]: {"a": 0.1}, DATES[1]: {"a": 0.2}})
    result = evaluate_weights(weights, labels, components=("returns",))
    assert result.tables["weights"].height == 1
    assert result.tables["returns"]["gross_return"].to_list() == [0.0, 0.2]


def test_stability_segments_enforce_economic_and_knowledge_dates():
    from bagelquant_bt.diagnostics import stability_tables
    from bagelquant_bt.horizon import PredictionHorizonDiagnostics

    ic = pl.DataFrame(
        {
            "evaluation_date": DATES[:2],
            "window_kind": ["cumulative"] * 2,
            "window_id": ["1D"] * 2,
            "start_session": [1] * 2,
            "end_session": [1] * 2,
            "target_start_date": DATES[1:3],
            "target_end_date": DATES[2:4],
            "available_date": [DATES[3], DATES[3]],
            "pearson_ic": [0.2, 0.4],
            "spearman_ic": [None, None],
        },
        schema_overrides={"spearman_ic": pl.Float64},
    )

    def diagnostics(source):
        return PredictionHorizonDiagnostics(
            **{
                name: source
                if name == "ic"
                else 0
                if name == "max_window_forward_rows"
                else pl.DataFrame()
                for name in PredictionHorizonDiagnostics.__dataclass_fields__
            }
        )

    periods = {
        "early": (DATES[0], DATES[2]),
        "known": (DATES[0], DATES[3]),
        "all": (DATES[0], DATES[4]),
    }
    stability, _ = stability_tables(
        diagnostics(ic), periods=periods, rolling_observations=2
    )
    early = stability.filter(
        (pl.col("segment") == "early") & (pl.col("metric") == "pearson_ic")
    )
    assert early.item(0, "sample_size") == 0
    assert early.item(0, "mean") is None
    assert early.item(0, "status") == "unavailable"
    assert early.item(0, "reason") == "no mature observations in requested segment"
    known = stability.filter(
        (pl.col("segment") == "known") & (pl.col("metric") == "pearson_ic")
    )
    assert known.item(0, "sample_size") == 2
    assert known.item(0, "mean") == pytest.approx(0.3)
    assert known.item(0, "available_date") == DATES[3]

    delayed = ic.with_columns(
        pl.when(pl.col("evaluation_date") == DATES[0])
        .then(pl.lit(DATES[4]))
        .otherwise(pl.col("available_date"))
        .alias("available_date")
    )
    stability, rolling = stability_tables(
        diagnostics(delayed), periods=periods, rolling_observations=2
    )
    known = stability.filter(
        (pl.col("segment") == "known") & (pl.col("metric") == "pearson_ic")
    )
    assert known.item(0, "sample_size") == 1
    assert known.item(0, "mean") == pytest.approx(0.4)
    assert known.item(0, "target_end_date") == DATES[3]
    null_metric = stability.filter(
        (pl.col("segment") == "all") & (pl.col("metric") == "spearman_ic")
    )
    assert null_metric.item(0, "reason") == (
        "no finite observations in requested segment"
    )
    pearson = rolling.filter(pl.col("metric") == "pearson_ic")
    assert pearson.item(0, "reason") == (
        "insufficient finite observations for rolling window"
    )
    assert pearson.item(1, "rolling_mean") == pytest.approx(0.3)
    assert pearson.item(1, "target_start_date") == DATES[1]
    assert pearson.item(1, "target_end_date") == DATES[3]
    assert pearson.item(1, "available_date") == DATES[4]
    assert slice_primitives({"rolling": pearson}, DATES[0], DATES[3])[
        "rolling"
    ].is_empty()
    assert (
        slice_primitives({"rolling": pearson}, DATES[0], DATES[4])["rolling"].height
        == 2
    )
    with pytest.raises(ValueError, match="require target_end_date and available_date"):
        stability_tables(diagnostics(ic.drop("available_date")), periods=periods)
