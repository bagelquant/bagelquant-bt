# Quick Start

`bagelquant-bt` evaluates research outputs. It does not retrieve data and it
does not build factor signals. Inputs are numeric long-form Polars DataFrames.

## Install

```bash
uv add bagelquant-bt
```

## Weight Backtest

Use `kind="weights"` when the signal frame already contains portfolio weights.
Weights use `time`, `asset_id`, and `weight`; prices use `time`, `asset_id`,
and `price`.

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

result.summary
result.net_cumulative_returns
```

## Factor Evaluation

Use `kind="factor"` when the first frame contains cross-sectional factor scores
with `time`, `asset_id`, and `factor` columns.
The package computes forward returns, information coefficients, quantile
returns, and a top-N backtest.

```python
from bagelquant_bt import BacktestConfig, run_backtest

factor = pl.DataFrame(
    {"time": ["2024-01-02"], "asset_id": ["AAA"], "factor": [1.5]}
)

result = run_backtest(
    factor,
    prices,
    kind="factor",
    config=BacktestConfig(
        initial_capital=1_000_000,
        quantiles=5,
        top_n=50,
    ),
)

result.ic_mean
result.top_minus_bottom
```

## Transaction Costs

```python
from bagelquant_bt import BacktestConfig, TransactionCostConfig

config = BacktestConfig(
    initial_capital=1_000_000,
    transaction_cost=TransactionCostConfig(rate=0.00015, min_fee=5.0),
)
```

Minimum fees require `initial_capital` so the engine can translate weight
turnover into traded notional.

