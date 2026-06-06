# Factor Evaluation

Factor evaluation treats the factor DataFrame as cross-sectional scores.

Higher scores are better.

## IC and ICIR

For each date, `bagelquant-bt` computes the cross-sectional correlation between
factor scores at date `t` and asset returns from `t` to `t+1`.

The default IC method is Spearman rank correlation:

```python
BacktestConfig(initial_capital=1_000_000, ic_method="spearman")
```

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
