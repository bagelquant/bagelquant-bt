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
weight or factor at date t -> earns close-to-close return from t to t+1
```

This means the final price date cannot produce a realized forward return and is
dropped from return calculations.

## Alignment

`bagelquant-bt` aligns prices and signal values by intersecting `(time, asset_id)` keys.

It rejects:

- duplicate `(time, asset_id)` keys
- nonnumeric values
- empty date overlap
- empty asset overlap
- non-DataFrame signal inputs
