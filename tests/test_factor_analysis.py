from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from bagelquant_bt.factor_analysis import (
    factor_return_correlation,
    holdings_factor_exposure,
    incremental_ic_summary,
    monthly_rank_returns,
    partial_rank_ic,
)


def test_rank_ic_duplicate_controls_and_no_independent_signal():
    rng = np.random.default_rng(4)
    x, z = rng.normal(size=(2, 200))
    y = x + z + rng.normal(size=200)
    single = partial_rank_ic(x, y, z[:, None])
    duplicate = partial_rank_ic(x, y, np.column_stack([z, z]))
    assert duplicate["ic"] == pytest.approx(single["ic"])
    assert duplicate["effective_rank"] == 2
    assert duplicate["ic"] > 0.4
    assert partial_rank_ic(z, y, z[:, None])["reason"] == "no_independent_signal"
    assert partial_rank_ic(np.ones(200), y)["reason"] == "no_independent_signal"


def test_rank_ic_complete_cases_and_empty_controls():
    x = np.arange(20, dtype=float)
    y = x.copy()
    y[0] = np.nan
    result = partial_rank_ic(x, y)
    assert result["method"] == "rank_ic"
    assert result["sample_count"] == 19
    assert result["ic"] == pytest.approx(1)
    assert partial_rank_ic(x[:3], y[:3])["reason"] == "insufficient_cross_section"


def test_correlation_pairwise_dates_and_constant():
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    left = pl.DataFrame({"time": dates, "value": list(range(20))})
    right = pl.DataFrame({"time": dates[2:], "value": [-i for i in range(2, 20)]})
    result = factor_return_correlation(left, right)
    assert result["sample_count"] == 18
    assert result["correlation"] == pytest.approx(-1)
    assert (
        factor_return_correlation(left.head(11), right)["reason"]
        == "insufficient_samples"
    )
    constant = left.with_columns(pl.lit(1).alias("value"))
    assert factor_return_correlation(left, constant)["reason"] == "constant_series"


def test_hac_inference_and_long_horizon():
    rng = np.random.default_rng(34)
    values = rng.normal(0.1, 0.2, 300)
    short = incremental_ic_summary(values.tolist())
    long = incremental_ic_summary(values.tolist(), horizon=60)
    assert long["lag"] == 59
    assert short["confidence_low"] < short["mean"] < short["confidence_high"]
    assert short["p_value"] < 0.05
    assert incremental_ic_summary(values[:11].tolist())["p_value"] is None
    assert incremental_ic_summary(values[:20].tolist(), horizon=60)["p_value"] is None


def test_hac_accounts_for_positive_serial_correlation():
    rng = np.random.default_rng(123)
    values = np.zeros(500)
    for i in range(1, len(values)):
        values[i] = 0.85 * values[i - 1] + rng.normal(scale=0.1)
    result = incremental_ic_summary(values.tolist())
    naive_error = np.std(values, ddof=0) / np.sqrt(len(values))
    assert result["standard_error"] > 1.5 * naive_error
    assert result["lag"] == int(4 * (len(values) / 100) ** (2 / 9))


@pytest.mark.parametrize("metric", ["book_return", "tail_return"])
def test_monthly_gross_one_and_missing_label_keeps_weight(metric):
    day = date(2024, 1, 31)
    factor = pl.DataFrame(
        {
            "evaluation_date": [day] * 10,
            "execution_date": [day] * 10,
            "asset_id": [str(i) for i in range(10)],
            "factor": list(range(10)),
        }
    )
    labels = factor.select("evaluation_date", "asset_id").with_columns(
        pl.Series("forward_return", [-0.1] * 5 + [0.1] * 5),
    )
    full = monthly_rank_returns(factor, labels, metric=metric).row(0, named=True)
    assert full["value"] == pytest.approx(0.1)
    missing = monthly_rank_returns(
        factor, labels.filter(pl.col("asset_id") != "9"), metric=metric
    ).row(0, named=True)
    assert missing["value"] < full["value"]
    assert missing["observed_count"] == missing["expected_count"] - 1
    if metric == "tail_return":
        assert missing["value"] == pytest.approx(0.05)


def test_holdings_exposure_actual_weight_cash_and_missing():
    signals = pl.DataFrame({"asset_id": ["a", "b", "c"], "value": [-1.0, 0.0, 1.0]})
    positions = pl.DataFrame({"asset_id": ["c"], "market_value": [50.0]})
    result = holdings_factor_exposure(signals, positions, equity=100)
    assert result["exposure"] == pytest.approx(0.5 / np.std([-1, 0, 1]))
    assert result["coverage_ratio"] == 1
    tiny_units = signals.with_columns(pl.col("value") * 1e-16)
    assert holdings_factor_exposure(tiny_units, positions, equity=100)[
        "exposure"
    ] == pytest.approx(result["exposure"])
    missing = positions.vstack(
        pl.DataFrame({"asset_id": ["unknown"], "market_value": [25.0]})
    )
    partial = holdings_factor_exposure(signals, missing, equity=100)
    assert partial["exposure"] == result["exposure"]
    assert partial["coverage_ratio"] == pytest.approx(2 / 3)
    assert not partial["complete"]
    assert (
        holdings_factor_exposure(signals, positions, equity=0)["reason"]
        == "invalid_equity"
    )
    assert (
        holdings_factor_exposure(signals, positions.head(0), equity=100)["exposure"]
        == 0
    )
