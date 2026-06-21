# Public API

The stable public API is exported from `bagelquant_bt`.

## Entry Points

```python
from bagelquant_bt import run_backtest, run_factor_evaluation, run_weight_backtest
```

- `run_backtest(signal, prices, *, kind, config=None)`: dispatch by `kind`.
- `run_weight_backtest(weights, prices, *, config)`: evaluate portfolio weights.
- `run_factor_evaluation(factor, prices, *, config)`: evaluate factor scores.
- `summary_report(result, *, output_path=None, missing_price_keys_output_path=None, title=None, annualization=252)`:
  build a static HTML report for `BacktestResult` or `FactorEvaluationResult`.
  Missing price keys are written to a separate CSV. With `output_path`, the
  default CSV path is `<report_stem>_missing_price_keys.csv`; pass
  `missing_price_keys_output_path` to override it or write only the CSV.

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
- `coverage`: per-price-date raw `weight_asset_count`, `universe_asset_count`,
  and `coverage_ratio`.
- `missing_price_keys`

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
- `coverage`: per-price-date raw `factor_signal_asset_count`,
  `universe_asset_count`, and `coverage_ratio`.
- `missing_price_keys`

## Visualization

`plot_coverage(result)` returns a Plotly time series showing raw factor-signal
or weight asset coverage as a shaded area and the total price-universe asset count.

## Exceptions

- `BagelQuantBacktestError`: base package error.
- `BacktestConfigError`: invalid configuration.
- `InputValidationError`: invalid or incompatible input frames.
