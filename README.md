# bagelquant-bt

`bagelquant-bt` provides backtesting, factor evaluation, performance metrics, and
visualization for the BagelQuant ecosystem.

The package is intentionally DataFrame-first:

- `bagelquant-core` builds and computes factors, signals, and portfolio weights.
- `bagelquant-data` retrieves and shapes datasets.
- `bagelquant-bt` evaluates already-computed DataFrames against daily prices.

Package code does not import `bagelquant-core` or `bagelquant-data`. The public
boundary is a numeric `pandas.DataFrame`.

## Portfolio Weight Backtest

```python
from bagelquant_bt import BacktestConfig, run_backtest

result = run_backtest(
    weights,
    prices,
    kind="weights",
    config=BacktestConfig(initial_capital=1_000_000),
)

result.summary.sharpe
result.gross_cumulative_returns
result.net_cumulative_returns
```

`prices` must be a daily close-price `DataFrame` indexed by date and columned by
asset. Weight values use the same shape. Weights on date `t` earn
close-to-close returns from `t` to `t+1`.

## Factor Evaluation

```python
from bagelquant_bt import BacktestConfig, run_backtest

result = run_backtest(
    factor,
    prices,
    kind="factor",
    config=BacktestConfig(initial_capital=1_000_000, top_n=50),
)

result.ic_mean
result.icir
result.quantile_cumulative_returns
result.top_n_backtest.summary.total_return
```

Factor evaluation computes daily cross-sectional IC, ICIR, quantile portfolio
returns, top-minus-bottom spread, and a long-only TOP N equal-weight portfolio
backtest.

## Transaction Costs

The default transaction cost model is:

```python
TransactionCostConfig(rate=0.00015, min_fee=5.0)
```

For each asset with nonzero traded notional:

- raw fee is `traded_notional * rate`
- applied fee is `max(raw_fee, min_fee)`
- daily cost return is total fee divided by portfolio value before trading

`BacktestConfig(initial_capital=...)` is required because the minimum fee is a
currency amount. Every backtest includes both gross no-cost and net cost-adjusted
returns.

## Visualization

Visualization helpers return matplotlib figure objects and never call
`plt.show()`:

```python
from bagelquant_bt.visualization import plot_cumulative_returns, plot_drawdown

fig, ax = plot_cumulative_returns(result)
fig, ax = plot_drawdown(result)
```

## Documentation

- [Quick start](docs/en/quick-start.md)
- [Concepts](docs/en/concepts.md)
- [Architecture](docs/en/architecture.md)
- [API](docs/en/reference/api.md)
- [Public API](docs/en/reference/public-api.md)
- [Transaction costs](docs/en/reference/transaction-costs.md)
- [Factor evaluation](docs/en/reference/factor-evaluation.md)

## Examples

Run examples from the package root:

```bash
uv run python examples/weight_backtest.py
uv run python examples/factor_evaluation.py
uv run python examples/visualization.py
```

The visualization example writes `cumulative_returns.png` and `drawdown.png`.
