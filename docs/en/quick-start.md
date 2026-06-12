# Quick Start

`bagelquant-bt` evaluates research outputs. It does not retrieve data and it
does not build factor signals. Inputs are numeric pandas DataFrames.

## Install

```bash
uv add bagelquant-bt
```

## Weight Backtest

Use `kind="weights"` when the signal frame already contains portfolio weights.
Rows are dates, columns are assets, and values are target weights.

```python
from bagelquant_bt import BacktestConfig, run_backtest

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

Use `kind="factor"` when the first frame contains cross-sectional factor scores.
The package computes forward returns, information coefficients, quantile
returns, and a top-N backtest.

```python
from bagelquant_bt import BacktestConfig, run_backtest

result = run_backtest(
    factor_scores,
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

