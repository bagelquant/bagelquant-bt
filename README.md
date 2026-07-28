# bagelquant-bt

`bagelquant-bt` provides backtesting, factor evaluation, performance metrics, and
Plotly visualization for long-form Polars panels.

The public input shape is always explicit:

- prices: `time`, `asset_id`, `price`
- weights: `time`, `asset_id`, `weight`
- factors: `time`, `asset_id`, `factor`

Weights at `time=t` earn the next market-session return. During an asset-specific
price gap, its holding is frozen at the last observed price; the gap sessions
have zero return and the cumulative move is recognized when a new price arrives.

```python
import polars as pl

from bagelquant_bt import BacktestConfig, run_backtest

prices = pl.DataFrame(
    {
        "time": ["2024-01-02", "2024-01-03"],
        "asset_id": ["AAA", "AAA"],
        "price": [100.0, 102.0],
    }
)
weights = pl.DataFrame(
    {"time": ["2024-01-02"], "asset_id": ["AAA"], "weight": [1.0]}
)

result = run_backtest(
    weights,
    prices,
    kind="weights",
    config=BacktestConfig(initial_capital=1_000_000),
)

print(result.returns)
print(result.summary)
```

Factor evaluation computes cross-sectional IC at the signal cadence, plus
daily marked-to-market quantile portfolios, a `q1 - qN` spread, and a TOP N
equal-weight backtest. Sparse or monthly signal snapshots rebalance only on
their snapshot dates; their weights are held across intervening daily returns.
Call `SignalPolicy.select` first and pass its `SignalSelection` to every
evaluation entry point; `ExecutionPolicy("next_open")` keeps date selection
separate from execution timing.

Visualization helpers return Plotly figures:

```python
from bagelquant_bt.visualization import plot_coverage, plot_cumulative_returns

fig = plot_cumulative_returns(result)
fig.write_html("cumulative_returns.html")

coverage_fig = plot_coverage(result)
coverage_fig.write_html("coverage.html")
```

## Development

```bash
uv run ruff check .
uv run pytest
```
