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
- `performance`
- `coverage`

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
- `coverage`

## `summary_report`

```python
summary_report(
    result,
    *,
    output_path=None,
    missing_price_keys_output_path=None,
    title=None,
    annualization=252,
)
```

Builds a static HTML report for `BacktestResult` or `FactorEvaluationResult`.
The report includes compact summary tables and Plotly figures. If `output_path`
is provided, the HTML is written to disk and also returned. Missing price keys
are written to a separate CSV instead of being embedded in the HTML report. By
default, the CSV is written next to the HTML as
`<report_stem>_missing_price_keys.csv`; pass `missing_price_keys_output_path` to
choose a different CSV path or to write the CSV when no HTML `output_path` is
provided.

Factor reports are grouped into IC and ICIR, TOP N, spread performance, and
quantile performance sections. Each section shows compact tables before plots.
Both factor and backtest reports show a coverage chart directly below their
top summary tables.

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

`ic_method` is accepted for compatibility. Factor evaluation now outputs both
Spearman and Pearson IC.

## DataFrame Boundary

The first argument must be a numeric `polars.DataFrame`.

Weights require `time`, `asset_id`, and `weight` columns. Factors require
`time`, `asset_id`, and `factor` columns. Prices require `time`, `asset_id`,
and `price` columns.
