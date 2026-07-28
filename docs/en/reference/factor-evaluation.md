# Factor Evaluation

Factor evaluation treats the factor DataFrame as cross-sectional scores.

Higher scores are better.

## IC and ICIR

For each date, `bagelquant-bt` computes the cross-sectional correlation between
factor scores at date `t` and asset returns from `t` to `t+1`.

Factor evaluation outputs both Pearson correlation and Spearman rank
correlation in `result.ic`:

```python
result.ic.select("time", "pearson_ic", "spearman_ic")
```

`result.ic_summary` includes mean, standard deviation, and ICIR for each method.
The compatibility fields `ic_mean`, `ic_std`, and `icir` use Spearman IC.

`icir` is:

```text
mean(IC) / standard_deviation(IC)
```

## Signal policies

`SignalPolicy.select` chooses whole snapshots from a daily prediction panel and
returns a `SignalSelection` containing both the resolved schedule and executable
rows. `month_end` prefers the last open session, falls back only to an earlier
whole snapshot in the same calendar month, and otherwise records a skip. It
never fills one asset from an earlier date.

`ExecutionPolicy("next_open")` separately maps rebalance dates to execution
dates. `run_signal_evaluation` accepts the resolved `SignalSelection`, measures
IC through the next execution date, and marks portfolios to market daily
between executions. The final signal without a complete following execution
period is excluded.

## Quantile Returns

Each day, assets are sorted by factor score from highest to lowest and split
into quantiles: `q1` contains the highest scores and `qN` the lowest.

Each quantile return is the equal-weight average forward return of assets in
that bucket.

The spread is:

```text
q1_return - qN_return
```

## TOP N Backtest

The TOP N backtest converts factor scores into long-only equal weights:

```text
top N assets each day -> 1 / N weight each
```

The resulting weight frame is passed through the same backtest engine as a
normal portfolio-weight DataFrame, including transaction costs.

## Spread and Lag Analysis

Factor evaluation also builds a spread portfolio: long `q1`, short `qN`, and
passes it through the same cost-aware backtest engine.

`lag_analysis` evaluates TOP N and spread portfolios with factor signals
delayed by 0, 1, 2, 3, 4, 5, 10, 20, 30, and 60 trading sessions. The delay is
resolved on the daily price-session calendar, so a monthly signal delayed by 15
is delayed by 15 open sessions rather than 15 monthly observations.

`lag_returns` contains gross and net cumulative return time series for the same
portfolio and lag combinations.

`ic_decay` reports mean Pearson and Spearman IC at the same trading-session
lags and is plotted as an IC decay line chart.
