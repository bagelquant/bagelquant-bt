# Signal Evaluation

Signal evaluation treats each scheduled `PredictionPanel` snapshot as
cross-sectional predictions. Higher values are better.

## Fixed prediction horizons

`run_prediction_horizon_diagnostics` is the prediction-only entry point. It
keeps the prediction update cadence separate from the economic forecast
horizon. The fixed daily protocol evaluates cumulative `1/5/10/20/40/60/120D`
returns and the disjoint `1D`, `2–5D`, `6–20D`, `21–60D`, and `61–120D`
buckets from the mapped next-open execution price. An unfinished end label is
excluded and reported through per-window coverage; the signal cross-section is
never selected or renormalized using future price availability.

The main continuous cross-sectional return uses average-tie percentile ranks,
centers them within the evaluation-date Universe, and normalizes by the sum of
absolute centered scores. The resulting Book has net zero, gross one, and
long/short books of `+0.5/-0.5`. `centered_rank_book_weights` returns unavailable
for a constant or one-sided cross-section. `gross_one_tail_weights` retains q1
long and q10 short at the same gross-one scale. Book, Tail, and the complete
quantile curve are diagnostics; they do not create NAV, costs, turnover,
Sharpe, or drawdown.

Every window reports Pearson/Spearman IC, positive-IC ratio, ICIR, Book, Tail,
quantile-rank IC, and cross-sectional regression slopes. Mean inference uses a
Bartlett Newey–West estimator with
`max(window_width-1, floor(4*(n/100)^(2/9)))` lags, two-sided p-values and 95%
confidence intervals. Benjamini–Hochberg q-values are calculated within each
metric family across all twelve windows. All modulo-window-width staggered
non-overlapping cohorts are retained as robustness diagnostics. Signal rank
persistence is evaluated at `1/5/10/20/40/60/120D`; half-life is reported only
as the first grid band crossing 0.5, or `>120D`.

The legacy `run_prediction_evaluation` sections below remain the portfolio
evaluation API. They are distinct from fixed-horizon prediction diagnostics.

## Daily diagnostic rank paths

`run_daily_rank_path_diagnostics` turns each PIT daily signal into the same
centered-rank Book and gross-one Tail targets, then runs them through the public
portfolio engine. Its output contains daily Book/Tail gross and net returns,
Book requested and executed turnover, and Book gross/net returns for every
integer lead or lag from `-30` through `+30`. Initial construction counts toward
turnover and cost and is explicitly marked by `is_initial_rebalance`, allowing
read-time summaries to exclude that one event without dropping the first row of
an arbitrary date slice. `alpha_return_lag_returns` additionally contains Book
and Tail gross/net paths at lags `0/1/2/5/10/20/60`; all fourteen paths use one
common complete date sample. Negative lags intentionally trade before the signal and are
therefore labeled non-PIT/look-ahead diagnostics; every lag uses the common
complete overlap of shifted execution dates.

The daily Summary autocorrelation grid uses average-tie cross-sectional rank
correlation for all lags `1..120`. Per-lag implied half-life is
`-lag * ln(2) / ln(rho_lag)` only when `0 < rho_lag < 1`. Rolling IC uses exactly
240 valid observations, is causal, and emits no value before that warm-up is
complete. These outputs support diagnostic charts and do not turn an Alpha
result into an account or Weight-Policy Portfolio Performance section.

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
