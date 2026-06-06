import pandas as pd

from bagelquant_bt import BacktestConfig, run_backtest

dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
prices = pd.DataFrame(
    {
        "AAA": [100.0, 104.0, 105.0, 106.0],
        "BBB": [100.0, 99.0, 101.0, 100.0],
        "CCC": [100.0, 101.0, 100.0, 103.0],
        "DDD": [100.0, 98.0, 99.0, 101.0],
    },
    index=dates,
)
factor = pd.DataFrame(
    {
        "AAA": [4.0, 2.0, 3.0, 1.0],
        "BBB": [1.0, 3.0, 2.0, 3.0],
        "CCC": [3.0, 4.0, 4.0, 4.0],
        "DDD": [2.0, 1.0, 1.0, 2.0],
    },
    index=dates,
)

result = run_backtest(
    factor,
    prices,
    kind="factor",
    config=BacktestConfig(initial_capital=1_000_000, quantiles=2, top_n=2),
)

print("IC:")
print(result.ic.round(4))
print(f"\nIC mean: {result.ic_mean:.4f}")
print(f"ICIR: {result.icir:.4f}")
print("\nQuantile returns:")
print(result.quantile_returns.round(6))
print("\nTOP N net cumulative return:")
print(result.top_n_backtest.net_cumulative_returns.round(6))
