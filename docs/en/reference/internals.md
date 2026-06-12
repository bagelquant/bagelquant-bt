# Internal Documentation

This page describes implementation details for maintainers.

## Weight Backtest Flow

```text
validate weights and prices
    |
    v
align to common dates/assets
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

Weights are interpreted as target weights at each timestamp. The cost model
uses weight changes and capital to estimate traded notional.

## Factor Evaluation Flow

```text
validate factor and prices
    |
    v
compute forward returns
    |
    v
compute cross-sectional IC
    |
    v
compute quantile returns
    |
    v
build top-N weights
    |
    v
run nested weight backtest
```

The nested top-N backtest reuses the same weight backtest engine, which keeps
cost and performance behavior consistent.

## Validation Principles

Validation should fail early for non-DataFrame inputs, duplicate axes,
non-numeric values, empty intersections, invalid config values, and unsupported
dispatch kinds. Error messages should name the offending input.

## Visualization

Plotting helpers should consume result objects and return matplotlib objects.
They should not recompute performance metrics or mutate result data.

