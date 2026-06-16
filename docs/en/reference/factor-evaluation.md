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

## Quantile Returns

Each day, assets are sorted by factor score and split into quantiles.

Each quantile return is the equal-weight average forward return of assets in
that bucket.

The top-minus-bottom spread is:

```text
highest_quantile_return - lowest_quantile_return
```

## TOP N Backtest

The TOP N backtest converts factor scores into long-only equal weights:

```text
top N assets each day -> 1 / N weight each
```

The resulting weight frame is passed through the same backtest engine as a
normal portfolio-weight DataFrame, including transaction costs.

## Long-Short and Lag Analysis

Factor evaluation also builds a top-quantile long, bottom-quantile short
portfolio and passes it through the same cost-aware backtest engine.

`lag_analysis` evaluates TOP N and long-short portfolios with factor signals
lagged by 0, 1, 2, 3, 4, 5, 10, 20, 30, and 60 observations.

`lag_returns` contains gross and net cumulative return time series for the same
portfolio and lag combinations.

`ic_decay` reports mean Pearson and Spearman IC at the same lags and is plotted
as an IC decay line chart.
