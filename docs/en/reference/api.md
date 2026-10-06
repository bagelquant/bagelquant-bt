# API

## `compose_prediction`

```python
compose_prediction(
    alpha_values,
    operator,
    calendar,
    alpha_policy,
    *,
    standardize_policy="none",
    execution_policy="next_open",
    prices=None,
)
```

`alpha_values` maps stable aliases to ordinary core `Node` values. `AlphaPolicy`
aligns evaluation-date snapshots; the independent `StandardizePolicy` then
applies `none`, `z_score`, or `percentile_rank`. Z-score standardization uses a
fixed asset order and per-date reduction, preserving exact historical bytes when
the future horizon or Arrow chunk layout changes. The result is the Operator's raw typed `Node`; BT
does not apply a fixed post-operator normalization. IC-weighted, OLS, and GLS
operators require prices so the package can construct
execution-to-next-execution labels without look-ahead.

## `run_prediction_backtest`

```python
run_prediction_backtest(
    prediction,
    prices,
    calendar,
    *,
    weight_policy,
    execution_policy="next_open",
    weight_inputs=None,
    config,
    execution_availability=None,
    slippage_rates=None,
)
```

The public backtest boundary requires a `Node`. `WeightPolicy`
creates evaluation-date target weights, then `ExecutionPolicy` maps those
weights to executable dates before invoking the private weight engine. Raw
DataFrames, ordinary `Node` instances, and direct weight frames raise a type
error.

Market-rule availability is supplied as a sparse frame with `time`,
`asset_id`, `can_buy`, `can_sell`, and `reason`. Missing rows are tradable. A
blocked change retains the prior executed weight and is retried until a newer
target supersedes it.

## `run_prediction_evaluation`

```python
run_prediction_evaluation(scheduled_signal, prices, *, config, ...)
```

The input is a `ScheduledPrediction`, normally produced by
`AlphaPolicy.select`. Forward returns run from the current execution price
to the next signal execution price. The result includes Spearman and Pearson
IC, quantiles, spread, TOP N, lag and IC-decay diagnostics, benchmarks, and
coverage.

## Data boundaries

Predictions use core `Node(time, asset_id, value)`. Prices remain
long-form Polars data with `time`, `asset_id`, and `price`. Weight policies emit an
ordinary weights `Node`; weights are deliberately not a public entry point.
