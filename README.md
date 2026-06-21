# bagelquant-bt

`bagelquant-bt` provides backtesting, factor evaluation, performance metrics, and
Plotly visualization for long-form Polars panels.

The public input shape is always explicit:

- prices: `time`, `asset_id`, `price`
- weights: `time`, `asset_id`, `weight`
- factors: `time`, `asset_id`, `factor`

Weights at `time=t` earn each asset's close-to-close return from `t` to the next
available observation.

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

Factor evaluation computes daily cross-sectional IC, quantile returns, a
top-minus-bottom spread, and a TOP N equal-weight backtest.

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
