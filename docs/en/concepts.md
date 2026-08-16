# Concepts

## Responsibility Boundary

`bagelquant-bt` evaluates a typed Signal against prices and applies investment
timing and portfolio policies.

- `bagelquant-core` owns `Panel`, `PredictionPanel`, signal construction, and research logic.
- `bagelquant-data` owns data access and storage.
- `bagelquant-bt` owns evaluation, transaction costs, summaries, and plots.

The public backtest boundary accepts `PredictionPanel` only. A plain `Panel`, raw
Polars `DataFrame`, or direct weights cannot enter `run_prediction_backtest`.

## DataFrame Shape

Prices, Signals, and weights share the same long-form key shape:

```text
keys:    time, asset_id
values:  price, value, or weight
```

`AlphaPolicy` maps observation snapshots, `ExecutionPolicy` maps execution
sessions, and `WeightPolicy.build(ScheduledPrediction)` returns a `WeightBuild`
containing a plain weights `Panel` plus skipped rows. `MarketRule` remains an
independent execution constraint. Lot sizing and live order submission are
outside the package boundary.

## Timing Convention

The package uses a no-lookahead convention:

```text
Signal information at date t -> executes only after its information cutoff
executed portfolio weights -> earn the next market-session close-to-close return
```

Signal and weight rows without an exact `(time, asset_id)` price key are dropped
from execution and listed in `missing_price_keys`. The final price date cannot
produce a realized forward return, so rows there may be retained as inputs but
will not contribute realized returns.

## Alignment

`bagelquant-bt` aligns signal and weight snapshots to exact price keys before
evaluation. Required null and NaN values are removed before alignment.

For portfolio weights, each timestamp is a complete target portfolio. The
backtest engine carries that target forward across price returns until the next
target portfolio arrives. Assets omitted from a later target become zero weight.
Turnover and transaction costs are calculated from these actual target-weight
changes, so holding days with unchanged weights do not create costs.

The daily valuation calendar is the union of observed price dates. When one
asset has no actual price on a market session, an existing holding is frozen at
its last observed price and earns zero for that session. Its cumulative price
move is recognized only when an actual price resumes. A rebalance can change an
asset's weight only on a date with its actual price: blocked existing positions
retain their prior weight, blocked new positions are not opened, and the
unexecuted target remains cash rather than being redistributed. Results expose
`price_gaps` and `unexecuted_weight_keys` for audit; `missing_price_keys`
continues to describe raw target keys without exact observed prices.

For signal evaluation, signal values remain inputs for analytics such as
IC, ICIR, and IC decay. Tradable factor outputs, including TOP N, quantile, and
spread portfolios, are first converted into portfolio weights and then sent
through the same weight backtest engine.

It rejects:

- duplicate `(time, asset_id)` keys
- nonnumeric values
- non-PredictionPanel public backtest inputs
