from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from bagelquant_bt.window import compute_window_tables


def _dates(count: int) -> list[date]:
    start = date(2024, 1, 1)
    return [start + timedelta(days=offset) for offset in range(count)]


def _returns(gross: list[float], net: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"time": _dates(len(gross)), "gross_return": gross, "net_return": net}
    )


def _spread_series(gross: list[float], net: list[float]) -> dict[str, pl.DataFrame]:
    return {
        "lag_returns": pl.DataFrame(
            {
                "lag": [0] * len(gross),
                "portfolio": ["spread"] * len(gross),
                "time": _dates(len(gross)),
                "gross_return": gross,
                "net_return": net,
            }
        ),
        "coverage": pl.DataFrame(
            {"time": _dates(len(gross)), "coverage_ratio": [0.8] * len(gross)}
        ),
    }


def test_summary_reports_window_risk_turnover_and_cost_drag() -> None:
    gross = [0.10, -0.05, 0.02, 0.03]
    net = [0.08, -0.06, 0.01, 0.02]
    metrics, tables = compute_window_tables(
        "summary",
        ("summary_table", "coverage"),
        returns=_returns(gross, net),
        turnover=pl.DataFrame(
            {"time": _dates(4), "turnover": [0.2, 0.0, 0.1, 0.0]}
        ),
        costs=pl.DataFrame(),
        series=_spread_series([0.04, -0.02, 0.01, 0.0], [0.03, -0.03, 0.0, 0.0]),
        annualization=4,
        ic_annualization=4,
    )

    net_values = np.asarray(net)
    expected_return = float(np.prod(1.0 + net_values) - 1.0)
    expected_volatility = float(net_values.std(ddof=1) * math.sqrt(4))
    wealth = np.cumprod(1.0 + net_values)
    peaks = np.maximum.accumulate(np.maximum(wealth, 1.0))
    expected_drawdown = float(np.min(wealth / peaks - 1.0))
    gross_return = float(np.prod(1.0 + np.asarray(gross)) - 1.0)

    assert metrics["top_n_net_annualized_return"] == pytest.approx(expected_return)
    assert metrics["top_n_net_annualized_volatility"] == pytest.approx(
        expected_volatility
    )
    assert metrics["top_n_net_max_drawdown"] == pytest.approx(expected_drawdown)
    assert metrics["top_n_net_calmar"] == pytest.approx(
        expected_return / abs(expected_drawdown)
    )
    assert metrics["top_n_annualized_turnover"] == pytest.approx(0.3)
    assert metrics["top_n_annualized_cost_drag"] == pytest.approx(
        gross_return - expected_return
    )
    assert metrics["spread_net_annualized_volatility"] is not None
    assert metrics["spread_net_max_drawdown"] == pytest.approx(-0.03)
    assert tables["summary"].row(0, named=True) == metrics


def test_summary_reports_ic_p_values_and_relative_wealth_excess() -> None:
    dates = _dates(4)
    series = _spread_series([0.01] * 4, [0.01] * 4)
    series["ic"] = pl.DataFrame(
        {
            "time": dates,
            "pearson_ic": [0.10, 0.20, 0.30, 0.40],
            "spearman_ic": [0.20, 0.25, 0.30, 0.35],
        }
    )
    top_n = _returns([0.03, 0.01, -0.01, 0.02], [0.02, 0.0, -0.02, 0.01])
    benchmark = pl.DataFrame(
        {
            "time": dates,
            "benchmark": ["selected"] * 4,
            "return": [0.01, -0.01, 0.0, 0.005],
        }
    )

    metrics, _ = compute_window_tables(
        "summary",
        ("summary_table", "coverage"),
        returns=top_n,
        turnover=pl.DataFrame(),
        costs=pl.DataFrame(),
        series=series,
        annualization=8,
        ic_annualization=4,
        benchmark_returns=benchmark,
    )

    portfolio_wealth = float(np.prod(1.0 + np.asarray([0.02, 0.0, -0.02, 0.01])))
    benchmark_wealth = float(np.prod(1.0 + np.asarray([0.01, -0.01, 0.0, 0.005])))
    assert metrics["top_n_net_annualized_excess_return"] == pytest.approx(
        (portfolio_wealth / benchmark_wealth) ** 2 - 1.0
    )
    assert metrics["pearson_ic_p_value"] == pytest.approx(0.0304662917)
    assert metrics["spearman_ic_p_value"] == pytest.approx(0.0033957771)


def test_summary_leaves_excess_unavailable_without_one_complete_benchmark() -> None:
    returns = _returns([0.01, 0.02], [0.01, 0.02])
    incomplete = pl.DataFrame(
        {
            "time": _dates(1),
            "benchmark": ["selected"],
            "return": [0.0],
        }
    )

    for benchmark in (None, incomplete):
        metrics, _ = compute_window_tables(
            "summary",
            ("summary_table",),
            returns=returns,
            turnover=pl.DataFrame(),
            costs=pl.DataFrame(),
            series=_spread_series([0.0, 0.0], [0.0, 0.0]),
            annualization=2,
            ic_annualization=2,
            benchmark_returns=benchmark,
        )
        assert metrics["top_n_net_annualized_excess_return"] is None
        assert metrics["top_n_net_annualized_return"] is not None


def test_summary_reports_negative_one_excess_for_zero_top_n_wealth() -> None:
    metrics, _ = compute_window_tables(
        "summary",
        ("summary_table",),
        returns=_returns([0.0, 0.0], [-1.0, 0.0]),
        turnover=pl.DataFrame(),
        costs=pl.DataFrame(),
        series=_spread_series([0.0, 0.0], [0.0, 0.0]),
        annualization=4,
        ic_annualization=4,
        benchmark_returns=pl.DataFrame(
            {
                "time": _dates(2),
                "benchmark": ["selected", "selected"],
                "return": [0.0, 0.0],
            }
        ),
    )

    assert metrics["top_n_net_annualized_excess_return"] == -1.0


def test_summary_handles_zero_drawdown_empty_and_non_finite_inputs() -> None:
    metrics, _ = compute_window_tables(
        "summary",
        ("summary_table",),
        returns=_returns([0.0, 0.0], [0.0, 0.0]),
        turnover=pl.DataFrame(
            {"time": _dates(2), "turnover": [0.2, float("nan")]}
        ),
        costs=pl.DataFrame(),
        series=_spread_series([0.0, 0.0], [0.0, 0.0]),
        annualization=2,
        ic_annualization=2,
    )

    assert metrics["top_n_net_annualized_volatility"] == 0.0
    assert metrics["top_n_net_sharpe"] is None
    assert metrics["top_n_net_max_drawdown"] == 0.0
    assert metrics["top_n_net_calmar"] is None
    assert metrics["top_n_annualized_turnover"] == pytest.approx(0.4)

    empty_metrics, _ = compute_window_tables(
        "summary",
        ("summary_table",),
        returns=pl.DataFrame(),
        turnover=pl.DataFrame(),
        costs=pl.DataFrame(),
        series={},
        annualization=2,
        ic_annualization=2,
    )
    assert empty_metrics["top_n_net_annualized_return"] is None
    assert empty_metrics["top_n_annualized_turnover"] is None
    assert empty_metrics["top_n_annualized_cost_drag"] is None


@pytest.mark.parametrize(
    ("section", "item", "table"),
    [
        ("spread", "spread_rolling_vol", "spread_rolling_vol"),
        ("top_n", "rolling_vol", "top_n_rolling_vol"),
    ],
)
def test_rolling_performance_contains_half_and_full_year_sharpe(
    section: str,
    item: str,
    table: str,
) -> None:
    gross = [0.10, 0.0, 0.10, 0.0]
    net = [0.08, -0.02, 0.08, -0.02]
    _, tables = compute_window_tables(
        section,
        (item,),
        returns=_returns(gross, net),
        turnover=pl.DataFrame(),
        costs=pl.DataFrame(),
        series=_spread_series(gross, net),
        annualization=4,
        ic_annualization=4,
    )

    rolling = tables[table]
    assert rolling.get_column("window").unique().sort().to_list() == [2, 4]
    half_year = rolling.filter(pl.col("window") == 2)
    full_year = rolling.filter(pl.col("window") == 4)
    assert half_year.get_column("gross_sharpe").drop_nulls()[0] == pytest.approx(
        math.sqrt(2)
    )
    assert full_year.get_column("gross_sharpe").drop_nulls()[0] == pytest.approx(
        math.sqrt(3)
    )
