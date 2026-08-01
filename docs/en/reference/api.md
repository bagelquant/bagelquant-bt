# API

## `compose_signal`

```python
compose_signal(
    alpha_values,
    composer,
    calendar,
    signal_date_policy,
    *,
    standardization="zscore",
    execution_policy="next_open",
    prices=None,
)
```

`alpha_values` maps stable aliases to ordinary core `Panel` values. The result
is a terminal `SignalPanel`. Standardization is mandatory and is either
`zscore` or `percentile_rank`. IC-weighted, OLS, and GLS composers require
prices so the package can construct execution-to-next-execution labels without
look-ahead.

## `run_signal_backtest`

```python
run_signal_backtest(
    signal,
    prices,
    calendar,
    signal_date_policy,
    *,
    execution_policy="next_open",
    portfolio_policy=None,
    portfolio_inputs=None,
    config,
    execution_availability=None,
)
```

The public backtest boundary requires a `SignalPanel`. It applies
`SignalDatePolicy`, `ExecutionPolicy`, and a portfolio policy before invoking
the private weight engine. Raw DataFrames, ordinary `Panel` instances, and
direct weight frames raise a type error.

Market-rule availability is supplied as a sparse frame with `time`,
`asset_id`, `can_buy`, `can_sell`, and `reason`. Missing rows are tradable. A
blocked change retains the prior executed weight and is retried until a newer
target supersedes it.

## `run_signal_evaluation`

```python
run_signal_evaluation(scheduled_signal, prices, *, config, ...)
```

The input is a `ScheduledSignal`, normally produced by
`SignalDatePolicy.select`. Forward returns run from the current execution price
to the next signal execution price. The result includes Spearman and Pearson
IC, quantiles, spread, TOP N, lag and IC-decay diagnostics, benchmarks, and
coverage.

## Data boundaries

Signals use core `SignalPanel(time, asset_id, value)`. Prices remain long-form
Polars data with `time`, `asset_id`, and `price`. Portfolio policies emit an
ordinary weights `Panel`; weights are deliberately not a public entry point.
