# API

## `compose_prediction`

```python
compose_prediction(
    alpha_values,
    composer,
    calendar,
    alpha_policy,
    *,
    standardize_policy="none",
    execution_policy="next_open",
    prices=None,
)
```

`alpha_values` maps stable aliases to ordinary core `Panel` values. `AlphaPolicy`
aligns evaluation-date snapshots; the independent `StandardizePolicy` then
applies `none`, `z_score`, or `percentile_rank`. The result is the Composer's raw typed `PredictionPanel`; BT
does not apply a fixed post-composer normalization. IC-weighted, OLS, and GLS
composers require prices so the package can construct
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

The public backtest boundary requires a `PredictionPanel`. `WeightPolicy`
creates evaluation-date target weights, then `ExecutionPolicy` maps those
weights to executable dates before invoking the private weight engine. Raw
DataFrames, ordinary `Panel` instances, and direct weight frames raise a type
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

Predictions use core `PredictionPanel(time, asset_id, value)`. Prices remain
long-form Polars data with `time`, `asset_id`, and `price`. Weight policies emit an
ordinary weights `Panel`; weights are deliberately not a public entry point.
