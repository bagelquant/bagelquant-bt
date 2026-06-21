# Internal Documentation

This page describes implementation details for maintainers.

## Weight Backtest Flow

```text
validate weights and prices
    |
    v
align target snapshots to tradable price dates
    |
    v
carry target weights forward across price returns
    |
    v
compute asset returns
    |
    v
compute gross portfolio returns
    |
    v
compute turnover and transaction costs
    |
    v
build BacktestResult
```

Weights are interpreted as complete target portfolios at each timestamp. If a
timestamp falls between price observations, it executes on the next available
tradable price date inside the covered range. The engine carries the latest
target weights forward until the next target arrives. The cost model uses actual
target-weight changes and capital to estimate traded notional.

## Factor Evaluation Flow

```text
validate factor and prices
    |
    v
align factor snapshots to tradable price dates
    |
    v
compute forward returns
    |
    v
compute cross-sectional IC, ICIR, and decay analytics
    |
    v
build factor-derived portfolio weights
    |
    v
run TOP N, quantile, spread, and lag portfolios through the weight backtest
```

Factor values remain signal inputs for analytics. Any tradable factor output is
converted into portfolio weights first, then routed through the same weight
backtest engine. This keeps holding periods, turnover, transaction costs, and
performance behavior consistent across direct weights, TOP N, quantile,
spread, lagged, and future portfolio styles.

## Validation Principles

Validation should fail early for non-Polars inputs, duplicate `(time, asset_id)`
keys, non-numeric values, timestamps outside price coverage, assets missing from
prices, invalid config values, and unsupported dispatch kinds. Error messages
should name the offending input.

## Visualization

Plotting helpers should consume result objects and return Plotly `go.Figure` objects.
They should not recompute performance metrics or mutate result data.
