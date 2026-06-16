from __future__ import annotations

import polars as pl

from bagelquant_bt import BacktestConfig, run_factor_evaluation


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
