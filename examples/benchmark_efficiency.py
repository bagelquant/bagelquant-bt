"""Local benchmark for large synthetic factor evaluations.

Run with:

    uv run python examples/benchmark_efficiency.py
"""

from __future__ import annotations

import datetime as dt
import time

import polars as pl

from bagelquant_bt import BacktestConfig, run_factor_evaluation


def main() -> None:
    assets = [f"a{index:04d}" for index in range(250)]
    start = dt.date(2024, 1, 1)
    dates = [
        start + dt.timedelta(days=offset)
        for offset in range(300)
    ]
    prices = pl.DataFrame(
        {
            "time": [date for asset in assets for date in dates],
            "asset_id": [asset for asset in assets for _ in dates],
            "price": [
                100.0 + asset_index * 0.01 + date_index * 0.1
                for asset_index, _ in enumerate(assets)
                for date_index, _ in enumerate(dates)
            ],
        }
    )
    factor = prices.select("time", "asset_id").with_columns(
        (
            (
                pl.col("asset_id").str.slice(1).cast(pl.Int64) * 13
                + pl.col("time").dt.ordinal_day() * 7
            )
            % 1_000
        )
        .cast(pl.Float64)
        .alias("factor")
    )

    start_time = time.perf_counter()
    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=1_000_000, quantiles=5, top_n=50),
    )
    elapsed = time.perf_counter() - start_time

    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"ic_rows={result.ic.height}")
    print(f"lag_analysis_rows={result.lag_analysis.height}")
    print(f"lag_returns_rows={result.lag_returns.height}")


if __name__ == "__main__":
    main()
