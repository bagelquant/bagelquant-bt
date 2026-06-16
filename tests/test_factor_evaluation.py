from __future__ import annotations

import polars as pl
import pytest

from bagelquant_bt import BacktestConfig, run_factor_evaluation
from bagelquant_bt.exceptions import InputValidationError


def test_factor_evaluation_uses_time_asset_id_inputs() -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "asset_id": ["a", "a", "b", "b"],
            "price": [1.0, 2.0, 2.0, 1.0],
        }
    )
    factor = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-01"],
            "asset_id": ["a", "b"],
            "factor": [2.0, 1.0],
        }
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )

    assert result.ic.height == 1
    assert result.ic.columns == ["time", "pearson_ic", "spearman_ic"]
    assert result.ic_summary["method"].to_list() == ["pearson", "spearman"]
    assert result.quantile_returns.select("time", "quantile", "return").height == 2
    assert result.top_n_weights.to_dicts()[0]["asset_id"] == "a"
    assert result.top_n_backtest.transaction_costs.data["total_fee"].sum() > 0
    assert result.top_n_backtest.performance.filter(
        pl.col("metric") == "sharpe"
    ).height == 1


def test_factor_evaluation_adds_long_short_and_lag_outputs() -> None:
    dates = [f"2024-01-{day:02d}" for day in range(1, 29)] + [
        f"2024-02-{day:02d}" for day in range(1, 29)
    ] + [f"2024-03-{day:02d}" for day in range(1, 11)]
    assets = ["a", "b", "c", "d"]
    prices = pl.DataFrame(
        {
            "time": [date for asset in assets for date in dates],
            "asset_id": [asset for asset in assets for _ in dates],
            "price": [
                10.0 + index * (0.2 if asset in {"a", "c"} else -0.1)
                for asset in assets
                for index, _ in enumerate(dates)
            ],
        }
    )
    factor = prices.select("time", "asset_id").with_columns(
        pl.when(pl.col("asset_id").is_in(["a", "c"]))
        .then(2.0)
        .otherwise(1.0)
        .alias("factor")
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )

    assert result.long_short_weights.height > 0
    assert result.long_short_backtest is not None
    assert result.long_short_backtest.transaction_costs.data["total_fee"].sum() > 0
    assert set(result.lag_analysis["portfolio"]) == {"top_n", "long_short"}
    expected_lags = {0, 1, 2, 3, 4, 5, 10, 20, 30, 60}
    assert set(result.lag_analysis["lag"]) == expected_lags
    assert set(result.lag_returns["lag"]) == expected_lags
    assert set(result.lag_returns["portfolio"]) == {"top_n", "long_short"}
    assert set(result.ic_decay["method"]) == {"pearson", "spearman"}


def test_low_frequency_factor_keeps_analytics_and_trades_daily_portfolios() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                date
                for asset in ["a", "b"]
                for date in [
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ],
            "asset_id": [asset for asset in ["a", "b"] for _ in range(4)],
            "price": [
                10.0,
                11.0,
                12.0,
                13.0,
                10.0,
                9.0,
                8.0,
                7.0,
            ],
        }
    )
    factor = pl.DataFrame(
        {
            "time": ["2024-01-02", "2024-01-02"],
            "asset_id": ["a", "b"],
            "factor": [2.0, 1.0],
        }
    )

    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
    )

    assert result.factor["time"].dt.strftime("%Y-%m-%d").unique().to_list() == [
        "2024-01-03"
    ]
    assert result.ic.height == 1
    assert result.icir == result.icir or result.ic_std != result.ic_std
    assert set(result.ic_decay["method"]) == {"pearson", "spearman"}
    assert result.top_n_backtest.returns["time"].dt.strftime("%Y-%m-%d").to_list() == [
        "2024-01-03",
        "2024-01-04",
    ]
    assert set(result.quantile_returns["quantile"]) == {"q1", "q2"}
    assert result.long_short_backtest is not None
    assert result.top_n_backtest.transaction_costs.data["total_fee"].to_list()[1] == 0.0


@pytest.mark.parametrize(
    ("factor", "match"),
    [
        (
            pl.DataFrame(
                {"time": ["2023-12-31"], "asset_id": ["a"], "factor": [1.0]}
            ),
            "outside covered price range",
        ),
        (
            pl.DataFrame(
                {"time": ["2024-01-03"], "asset_id": ["a"], "factor": [1.0]}
            ),
            "outside covered price range",
        ),
        (
            pl.DataFrame(
                {"time": ["2024-01-01"], "asset_id": ["b"], "factor": [1.0]}
            ),
            "assets missing from prices",
        ),
    ],
)
def test_factor_must_be_covered_by_prices(
    factor: pl.DataFrame,
    match: str,
) -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "asset_id": ["a", "a"],
            "price": [10.0, 11.0],
        }
    )

    with pytest.raises(InputValidationError, match=match):
        run_factor_evaluation(
            factor,
            prices,
            config=BacktestConfig(initial_capital=10_000, quantiles=2, top_n=1),
        )
