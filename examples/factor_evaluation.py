from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from bagelquant_bt import BacktestConfig, run_factor_evaluation
from bagelquant_bt.factor import FactorEvaluationResult


def make_prices(
    start: date = date(2024, 1, 1),
    end: date = date(2024, 12, 31),
    n_assets: int = 50,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic prices with geometric Brownian motion."""
    rng = np.random.default_rng(seed)

    dates = pl.DataFrame(
        {
            "time": pl.date_range(
                start=start,
                end=end,
                interval="1d",
                eager=True,
            )
        }
    )

    assets = pl.DataFrame(
        {
            "asset_id": [f"asset_{i:03d}" for i in range(n_assets)]
        }
    )

    panel = dates.join(assets, how="cross")

    n_rows = panel.height

    annual_mu = 0.08
    annual_sigma = 0.25
    dt = 1 / 252
    start_price = 100.0

    random_shocks = rng.normal(size=n_rows)

    prices = (
        panel
        .with_columns(
            shock=pl.Series(random_shocks)
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


def make_factor(
    prices: pl.DataFrame,
    seed: int = 123,
) -> pl.DataFrame:
    """Generate synthetic factor values for each date x asset."""
    rng = np.random.default_rng(seed)

    # Generate one random factor value for every date x asset.
    # Higher factor values are interpreted as better stocks.
    factor_values = rng.normal(
        loc=0.0,
        scale=1.0,
        size=prices.height,
    )

    factor = (
        prices
        .select("time", "asset_id")
        .with_columns(
            factor=pl.Series(factor_values)
        )
        .sort(["time", "asset_id"])
    )

    return factor


def main() -> None:
    prices = make_prices()
    factor = make_factor(prices)

    result: FactorEvaluationResult = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(
            initial_capital=1_000_000,
            quantiles=5,
            top_n=10,
        ),
    )

    print("IC:")
    print(result.ic.head())
    print(f"IC mean: {result.ic_mean}")
    print(f"ICIR: {result.icir}")

    print("\nQuantile returns:")
    print(result.quantile_returns.head())

    print("\nTOP N returns:")
    print(result.top_n_backtest.returns.head())


if __name__ == "__main__":
    main()
