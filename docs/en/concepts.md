# Concepts

## Responsibility Boundary

`bagelquant-bt` has one job: evaluate a research output against daily prices.

It does not import `bagelquant-core` or `bagelquant-data`.

- `bagelquant-core` owns signal construction and research logic.
- `bagelquant-data` owns data access and storage.
- `bagelquant-bt` owns evaluation, transaction costs, summaries, and plots.

This keeps the backtester useful with any workflow that can produce a
long-form Polars `DataFrame`.

## DataFrame Shape

Prices, weights, and factor scores use the same long-form key shape:

```text
keys:    time, asset_id
values:  price, weight, or factor
```

Prices are interpreted as close prices.

For a weight backtest, values are portfolio weights. Negative weights are
allowed.

For factor evaluation, values are cross-sectional scores. Higher scores
are considered better for quantile and TOP N tests.

## Timing Convention

The package uses a no-lookahead convention:

```text
weight or factor at date t -> executes on the next available price date
executed portfolio weights -> earn close-to-close return to the next price date
```

Signal and weight timestamps must be inside the covered price range. A timestamp
that falls between price observations is moved forward to the next available
tradable price date. The final price date cannot produce a realized forward
return, so it is not a tradable execution date.

## Alignment

`bagelquant-bt` aligns signal and weight snapshots to the price calendar before
evaluation.

For portfolio weights, each timestamp is a complete target portfolio. The
backtest engine carries that target forward across price returns until the next
target portfolio arrives. Assets omitted from a later target become zero weight.
Turnover and transaction costs are calculated from these actual target-weight
changes, so holding days with unchanged weights do not create costs.

For factor evaluation, factor values remain signal inputs for analytics such as
IC, ICIR, and IC decay. Tradable factor outputs, including TOP N, quantile, and
long-short portfolios, are first converted into portfolio weights and then sent
through the same weight backtest engine.

It rejects:

- duplicate `(time, asset_id)` keys
- nonnumeric values
- timestamps outside the covered price range
- assets missing from prices
- non-DataFrame signal inputs
