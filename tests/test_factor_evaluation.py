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
    assert result.quantile_returns.select("time", "quantile", "return").height == 2
    assert result.top_n_weights.to_dicts()[0]["asset_id"] == "a"
