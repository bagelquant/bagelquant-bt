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
run_weight_backtest(
    weights,
    prices,
    *,
    config,
    execution_availability=None,
)
```

Evaluates a long-form Polars DataFrame as portfolio weights.

`execution_availability` is an optional sparse table with `time`, `asset_id`,
`can_buy`, `can_sell`, and `reason`. Missing rows are tradable. A blocked
increment retains the prior executed weight and is retried on later price
sessions until executable; a new target supersedes the pending target.
`bagelquant-bt` never infers market rules from asset codes or exchanges. Costs
and turnover are recorded on actual execution dates, and blocked attempts are
available in `BacktestResult.execution_blocks`.

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
- `execution_blocks`

## `run_factor_evaluation`

```python
run_factor_evaluation(
    factor,
    prices,
    *,
    config,
    coverage_universe=None,
    benchmark_universe=None,
    execution_availability=None,
    benchmark_returns=None,
    benchmark_coverage=None,
)
```

Evaluates a long-form Polars DataFrame as factor scores.

The result always includes the cost-free `universe_equal_weight` benchmark.
`benchmark_universe` supplies its point-in-time membership independently of
factor coverage dates. Callers may append named benchmark returns and coverage;
the package does not load market or capitalization data.

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
- `spread_returns`
- `top_n_weights`
- `top_n_backtest`
- `spread_weights`
- `spread_backtest`
- `lag_analysis`
- `lag_returns`
- `ic_decay`
- `coverage`
- `benchmark_returns`
- `benchmark_coverage`
- `benchmark_performance`
- `excess_returns`

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

Factor reports are grouped into IC and ICIR, TOP N, TOP N versus benchmarks,
spread performance, and quantile performance sections. Benchmark excess output
contains daily return differences, compounded daily differences, and relative
wealth (`TOP N wealth / benchmark wealth - 1`) for both gross and net TOP N.
Each section shows compact tables before plots.
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
