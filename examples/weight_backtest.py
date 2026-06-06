import pandas as pd

from bagelquant_bt import BacktestConfig, run_backtest

dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
prices = pd.DataFrame(
    {
        "AAA": [100.0, 102.0, 101.0, 104.0],
        "BBB": [100.0, 99.0, 101.0, 102.0],
        "CCC": [100.0, 101.0, 103.0, 102.0],
    },
    index=dates,
)
weights = pd.DataFrame(
    {
        "AAA": [0.5, 0.2, 0.2, 0.4],
        "BBB": [0.3, 0.5, 0.2, 0.2],
        "CCC": [0.2, 0.3, 0.6, 0.4],
    },
    index=dates,
)

result = run_backtest(
    weights,
    prices,
    kind="weights",
    config=BacktestConfig(initial_capital=1_000_000),
)

print("Gross returns:")
print(result.gross_returns.round(6))
print("\nNet returns:")
print(result.net_returns.round(6))
print("\nSummary:")
print(result.summary)
