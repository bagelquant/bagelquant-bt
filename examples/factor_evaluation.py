import polars as pl

from bagelquant_bt import BacktestConfig, run_backtest

prices = pl.DataFrame(
    {
        "time": ["2024-01-02", "2024-01-03"] * 4,
        "asset_id": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC", "DDD", "DDD"],
        "price": [100.0, 104.0, 100.0, 99.0, 100.0, 101.0, 100.0, 98.0],
    }
)
factor = pl.DataFrame(
    {
        "time": ["2024-01-02"] * 4,
        "asset_id": ["AAA", "BBB", "CCC", "DDD"],
        "factor": [4.0, 1.0, 3.0, 2.0],
    }
)

result = run_backtest(
    factor,
    prices,
    kind="factor",
    config=BacktestConfig(initial_capital=1_000_000, quantiles=2, top_n=2),
)

print("IC:")
print(result.ic.to_dicts())
print("\nQuantile returns:")
print(result.quantile_returns.to_dicts())
print("\nTOP N returns:")
print(result.top_n_backtest.returns.to_dicts())
