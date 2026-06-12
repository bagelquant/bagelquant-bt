import polars as pl

from bagelquant_bt import BacktestConfig, run_backtest

prices = pl.DataFrame(
    {
        "time": ["2024-01-02", "2024-01-03", "2024-01-04"] * 3,
        "asset_id": ["AAA"] * 3 + ["BBB"] * 3 + ["CCC"] * 3,
        "price": [100.0, 102.0, 104.0, 100.0, 99.0, 102.0, 100.0, 101.0, 102.0],
    }
)
weights = pl.DataFrame(
    {
        "time": [
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
            "2024-01-03",
        ],
        "asset_id": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
        "weight": [0.5, 0.3, 0.2, 0.2, 0.5, 0.3],
    }
)

result = run_backtest(
    weights,
    prices,
    kind="weights",
    config=BacktestConfig(initial_capital=1_000_000),
)

print("Returns:")
print(result.returns.to_dicts())
print("\nSummary:")
print(result.summary)
