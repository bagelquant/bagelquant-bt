from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from bagelquant_bt import BacktestConfig, run_weight_backtest, summary_report


def make_prices(
    start: date = date(2024, 1, 1),
    end: date = date(2024, 12, 31),
    n_assets: int = 50,
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pl.DataFrame(
        {
            "time": pl.date_range(start, end, interval="1d", eager=True)
        }
    )

    assets = pl.DataFrame(
        {
            "asset_id": [f"asset_{i:03d}" for i in range(n_assets)]
        }
    )

    panel = dates.join(assets, how="cross")

    annual_mu = 0.08
    annual_sigma = 0.25
    dt = 1 / 252
    start_price = 100.0

    prices = (
        panel
        .with_columns(
            shock=pl.Series(rng.normal(size=panel.height))
        )
        .with_columns(
            log_return=(
                (annual_mu - 0.5 * annual_sigma**2) * dt
                + annual_sigma * np.sqrt(dt) * pl.col("shock")
            )
        )
        .sort(["asset_id", "time"])
        .with_columns(
            price=(
                start_price
                * pl.col("log_return").cum_sum().exp().over("asset_id")
            )
        )
        .select("time", "asset_id", "price")
        .sort(["time", "asset_id"])
    )

    return prices


def make_weights(
    prices: pl.DataFrame,
    seed: int = 123,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    monthly_dates = prices.select("time").unique().filter(pl.col("time").dt.day() == 1)
    assets = prices.select("asset_id").unique()
    monthly_panel = monthly_dates.join(assets, how="cross")

    weights = (
        monthly_panel
        .with_columns(
            raw_weight=pl.Series(
                rng.uniform(
                    low=0.0,
                    high=1.0,
                    size=monthly_panel.height,
                )
            )
        )
        .with_columns(
            weight=(
                pl.col("raw_weight")
                / pl.col("raw_weight").sum().over("time")
            )
        )
        .select("time", "asset_id", "weight")
        .sort(["time", "asset_id"])
    )

    return weights


def main() -> None:
    prices = make_prices()
    weights = make_weights(prices)

    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=1_000_000),
    )

    print("Prices:")
    print(prices.head())

    print("\nMonthly weights:")
    print(weights.head())

    print("\nDaily returns from held monthly weights:")
    print(result.returns.head())

    print("\nSummary:")
    print(result.summary)

    summary_report(result, output_path="backtest_summary_report.html")
    print("\nSaved backtest_summary_report.html")


if __name__ == "__main__":
    main()
