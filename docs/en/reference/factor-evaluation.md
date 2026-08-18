# Signal Evaluation

Signal evaluation treats each scheduled `PredictionPanel` snapshot as
cross-sectional predictions. Higher values are better.

## IC and ICIR

For each date, `bagelquant-bt` computes the cross-sectional correlation between
signal values at one execution date and asset returns through the next signal
execution date.

Signal evaluation outputs both Pearson correlation and Spearman rank
correlation in `result.ic`:

```python
result.ic.select("time", "pearson_ic", "spearman_ic")
```

`result.ic_summary` includes mean, standard deviation, and ICIR for each method.
The result-level `ic_mean`, `ic_std`, and `icir` fields report the configured
IC method.

`icir` is:

```text
mean(IC) / standard_deviation(IC) * sqrt(IC observations per year)
```

`BacktestConfig.ic_annualization` sets IC observations per year. When omitted,
it defaults to the daily-return `annualization` setting. This is distinct from
portfolio returns: annualized return, volatility, Sharpe, rolling performance,
benchmarks, and lag Sharpe use the daily-return annualization setting.

## Signal date and execution policies

`AlphaPolicy.select` chooses whole snapshots from a `PredictionPanel` and
returns `ScheduledPrediction`, which contains the resolved schedule, execution-date
lineage, and typed signal. `month_end` prefers the last open session, falls back
only to an earlier whole snapshot in the same calendar month, and otherwise
records a skip. It never fills one asset from an earlier date.

`ExecutionPolicy("next_open")` separately maps rebalance dates to execution
dates. `run_prediction_evaluation` accepts the resolved `ScheduledPrediction`, measures
IC through the next execution date, and marks portfolios to market daily
between executions. The final signal without a complete following execution
period is excluded.

For a monthly `month_end` policy, the value selected at a calendar month end is
therefore evaluated from its mapped next-open execution price through the next
scheduled execution price. In-progress months never enter IC or a rolling
supervised-composer window.

## Quantile Returns

Each day, assets are sorted by factor score from highest to lowest and split
into quantiles: `q1` contains the highest scores and `qN` the lowest.

Each quantile return is the equal-weight average forward return of assets in
that bucket.

The spread is:

```text
q1_return - qN_return
```

`quantile_rank_information_coefficients(quantile_returns, periods=...)` infers
N from the stored q1-to-qN labels, so historical q5 results remain readable.
When execution periods are supplied, it compounds each group's daily gross
returns inside `[time, next_time)` and emits exactly one observation at the
period start. It then assigns q1-to-qN scores N-to-1 and computes Spearman
correlation with the group returns. A strictly decreasing return path is `+1`,
a strictly increasing path is `-1`, and missing groups, groups without a finite
return, or equal group returns produce null. Result statistical tests use the
same complete periods and two-sided one-sample Student t-test as ordinary IC.

Quantile labels are always ordered by their numeric suffix in result tables,
figures, legends, and HTML reports: `q1, q2, ..., q10`, while historical q5
artifacts retain `q1, ..., q5`.

## TOP N Backtest

The TOP N backtest converts signal values into long-only equal weights:

```text
top N assets each day -> 1 / N weight each
```

The resulting weight frame is passed through the same backtest engine as a
normal portfolio-weight DataFrame, including transaction costs.

## Spread and Lag Analysis

Signal evaluation also builds a spread portfolio: long `q1`, short `qN`, and
passes it through the same cost-aware backtest engine.

`lag_analysis` evaluates TOP N and spread portfolios with signals
delayed by 0, 1, 2, 3, 4, 5, 10, 20, 30, and 60 trading sessions. The delay is
resolved on the daily price-session calendar, so a monthly signal delayed by 15
is delayed by 15 open sessions rather than 15 monthly observations.

`lag_returns` contains gross and net cumulative return time series for the same
portfolio and lag combinations.

`ic_decay` reports mean Pearson and Spearman IC at the same trading-session
lags and is plotted as an IC decay line chart.
