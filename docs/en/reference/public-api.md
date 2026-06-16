# Public API

The stable public API is exported from `bagelquant_bt`.

## Entry Points

```python
from bagelquant_bt import run_backtest, run_factor_evaluation, run_weight_backtest
```

- `run_backtest(signal, prices, *, kind, config=None)`: dispatch by `kind`.
- `run_weight_backtest(weights, prices, *, config)`: evaluate portfolio weights.
- `run_factor_evaluation(factor, prices, *, config)`: evaluate factor scores.
- `summary_report(result, *, output_path=None, title=None, annualization=252)`:
  build a static HTML report for `BacktestResult` or `FactorEvaluationResult`.

## Configuration

```python
from bagelquant_bt import BacktestConfig, TransactionCostConfig

config = BacktestConfig(
    initial_capital=1_000_000,
    transaction_cost=TransactionCostConfig(rate=0.00015, min_fee=5.0),
    annualization=252,
    ic_method="spearman",
    quantiles=5,
    top_n=50,
)
```

- `initial_capital` must be positive.
- `ic_method` is accepted for compatibility. Factor evaluation outputs both
  Spearman and Pearson IC.
- `quantiles` controls factor bucket count.
- `top_n` controls the top-N factor portfolio.

## Results

`BacktestResult` exposes:

- `weights`
- `asset_returns`
- `gross_returns`
- `net_returns`
- `gross_cumulative_returns`
- `net_cumulative_returns`
- `gross_value`
- `net_value`
- `turnover`
- `transaction_costs`
- `summary`
- `performance`

`FactorEvaluationResult` exposes:

- `factor`
- `forward_returns`
- `ic`
- `ic_summary`
- `ic_mean`
- `ic_std`
- `icir`
- `quantile_returns`
- `quantile_cumulative_returns`
- `top_minus_bottom`
- `top_n_weights`
- `top_n_backtest`
- `long_short_weights`
- `long_short_backtest`
- `lag_analysis`
- `lag_returns`
- `ic_decay`

## Exceptions

- `BagelQuantBacktestError`: base package error.
- `BacktestConfigError`: invalid configuration.
- `InputValidationError`: invalid or incompatible input frames.
