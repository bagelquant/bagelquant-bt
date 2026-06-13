# API

## `run_backtest`

```python
run_backtest(signal, prices, *, kind, config=None)
```

Dispatches to the correct evaluation path.

- `kind="weights"` calls `run_weight_backtest`
- `kind="factor"` calls `run_factor_evaluation`

`config` is required because transaction-cost minimum fees require
`initial_capital`.

## `run_weight_backtest`

```python
run_weight_backtest(weights, prices, *, config)
```

Evaluates a long-form Polars DataFrame as portfolio weights.

Returns `BacktestResult`.

Important fields:

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

## `run_factor_evaluation`

```python
run_factor_evaluation(factor, prices, *, config)
```

Evaluates a long-form Polars DataFrame as factor scores.

Returns `FactorEvaluationResult`.

Important fields:

- `factor`
- `forward_returns`
- `ic`
- `ic_mean`
- `ic_std`
- `icir`
- `quantile_returns`
- `quantile_cumulative_returns`
- `top_minus_bottom`
- `top_n_weights`
- `top_n_backtest`

## Config

```python
BacktestConfig(
    initial_capital=1_000_000,
    transaction_cost=TransactionCostConfig(rate=0.00015, min_fee=5.0),
    annualization=252,
    ic_method="spearman",
    quantiles=5,
    top_n=50,
)
```

`initial_capital` must be positive.

`ic_method` may be `"spearman"` or `"pearson"`.

## DataFrame Boundary

The first argument must be a numeric `polars.DataFrame`.

Weights require `time`, `asset_id`, and `weight` columns. Factors require
`time`, `asset_id`, and `factor` columns. Prices require `time`, `asset_id`,
and `price` columns.
