import math

import polars as pl
import pytest

from bagelquant_bt.performance import rolling_performance
from bagelquant_bt.statistics import (
    annualized_return,
    return_metrics,
    return_statistics,
)


def test_negative_terminal_wealth_has_no_real_annualized_return():
    metrics = return_metrics([-1.5, 0.0], 2)
    assert metrics["total_return"] == -1.5
    assert metrics["annualized_return"] is None
    assert metrics["annualized_return_reason"] == "terminal wealth must be nonnegative"
    assert metrics["calmar_reason"] == metrics["annualized_return_reason"]
    assert math.isnan(annualized_return(-0.5, periods=2, annualization=2))
    assert return_metrics([-1.0, 0.0], 2)["annualized_return"] == -1.0


def test_shared_return_statistics_excludes_missing_observations_and_first_loss():
    frame = pl.DataFrame(
        {"gross_return": [-0.1, None, 0.1], "net_return": [0.0, None, 0.0]}
    )
    metrics = return_statistics(frame, annualization=2)
    assert metrics["gross_sample_size"] == 2
    assert math.isclose(metrics["gross_max_drawdown"], -0.1)
    assert metrics["net_sharpe"] is None
    assert return_metrics([], 2)["total_return"] is None


@pytest.mark.parametrize("values", [[], [None, float("nan"), float("inf")]])
def test_empty_return_statistics_are_unavailable_with_reasons(values):
    metrics = return_metrics(values, 252)
    assert metrics["sample_size"] == 0
    assert metrics["status"] == "unavailable"
    assert metrics["reason"] == "no finite return observations"
    for name in (
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "hit_rate",
    ):
        assert metrics[name] is None
        assert metrics[f"{name}_reason"] == metrics["reason"]


def test_single_constant_and_complete_samples_explain_undefined_metrics():
    single = return_metrics([-0.01], 252)
    assert single["status"] == "partial"
    assert single["sharpe"] is None
    assert single["sharpe_reason"] == "at least two return observations required"
    assert single["annualized_volatility_reason"] == single["sharpe_reason"]
    constant = return_metrics([0.0, 0.0], 252)
    assert constant["annualized_volatility"] == 0.0
    assert constant["annualized_volatility_reason"] is None
    assert constant["sharpe_reason"] == "sample variance is zero"
    assert constant["calmar_reason"] == "maximum drawdown is zero"
    short = return_metrics([-0.02, 0.01, 0.03], 252)
    assert short["expected_shortfall_95"] is None
    assert short["expected_shortfall_95_reason"] == (
        "at least 20 finite observations required"
    )
    complete = return_metrics([-0.02, 0.01, 0.03] * 7, 252)
    assert complete["status"] == "complete"
    assert complete["reason"] is None
    assert all(
        value is None for name, value in complete.items() if name.endswith("_reason")
    )


def test_nonfinite_computed_statistics_are_null_with_reasons():
    metrics = return_metrics([1e308, 1e308], 252)
    assert metrics["total_return"] is None
    assert metrics["total_return_reason"] == "terminal wealth is non-finite"
    assert metrics["annualized_return"] is None
    assert metrics["annualized_return_reason"] == "annualized wealth is non-finite"
    assert metrics["max_drawdown"] is None
    assert metrics["max_drawdown_reason"] == "wealth path is non-finite"
    assert all(
        math.isfinite(value) for value in metrics.values() if isinstance(value, float)
    )


def test_rolling_statistics_preserve_unavailable_reason_string_columns():
    returns = pl.DataFrame(
        {
            "time": [1, 2, 3, 4],
            "gross_return": [0.0, 0.0, None, 0.01],
            "net_return": [0.0, 0.0, float("inf"), 0.01],
        }
    )
    rolling = rolling_performance(returns, annualization=252, windows=(1, 2))
    assert rolling.schema["gross_sharpe_reason"] == pl.String
    assert rolling.schema["net_volatility_reason"] == pl.String
    pair = rolling.filter(pl.col("window") == 2)
    assert pair.item(0, "gross_sharpe_reason") == (
        "insufficient finite observations for rolling window"
    )
    assert pair.item(1, "gross_volatility") == 0.0
    assert pair.item(1, "gross_volatility_reason") is None
    assert pair.item(1, "gross_sharpe_reason") == "sample variance is zero"
    assert pair.item(2, "net_sharpe") is None
    assert pair.item(2, "net_sharpe_reason") == (
        "insufficient finite observations for rolling window"
    )
    assert rolling.filter(pl.col("window") == 1).item(0, "gross_sharpe_reason") == (
        "at least two return observations required"
    )
