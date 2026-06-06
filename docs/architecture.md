# Architecture And Design

`bagelquant-bt` is a DataFrame-first evaluation package.

```text
prices + weights/factor scores
    |
    v
input validation
    |
    v
returns, turnover, costs, IC, quantiles
    |
    v
result dataclasses
    |
    v
visualization helpers
```

## Philosophy

- Keep research generation outside the backtester.
- Treat DataFrames as the public boundary.
- Make transaction costs explicit and reproducible.
- Return structured result objects instead of printing reports.
- Keep visualization as a thin layer over result objects.

## Structure

- `inputs`: frame validation, alignment, and numeric checks.
- `returns`: asset returns and cumulative return utilities.
- `costs`: turnover and transaction-cost calculations.
- `engine`: weight backtest orchestration and public dispatch.
- `factor`: information coefficient, quantile, and top-N evaluation.
- `performance`: summary metrics.
- `results`: dataclasses for downstream inspection.
- `visualization`: plotting helpers.

## Data Boundary

Input rows are dates and columns are assets. Weight frames contain target
weights. Factor frames contain cross-sectional scores. Price frames contain
numeric prices used to compute returns.

The package does not import `bagelquant-data` or `bagelquant-core`; callers
adapt those outputs into DataFrames before evaluation.

