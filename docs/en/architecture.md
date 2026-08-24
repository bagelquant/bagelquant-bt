# Architecture And Design

`bagelquant-bt` is a typed-signal evaluation package.

```text
AlphaValue Panel + prices + policies
    |
    v
PredictionComposer -> PredictionPanel
    |
    v
schedule -> execution -> portfolio weights
    |
    v
returns, turnover, costs, IC, quantiles
    |
    v
visualization helpers
```

## Philosophy

- Keep Alpha definitions and data retrieval outside the backtester.
- Require `PredictionPanel` at the public backtest boundary.
- Make transaction costs explicit and reproducible.
- Return structured result objects instead of printing reports.
- Keep visualization as a thin layer over result objects.
- Keep exchange-specific execution rules caller-authored and opt-in. The core
  consumes a generic availability table and never guesses a market from codes.
- Build only the universe equal-weight benchmark internally; capitalization
  and index benchmarks remain caller-provided data.

## Structure

- `inputs`: frame validation, alignment, and numeric checks.
- `returns`: asset returns and cumulative return utilities.
- `costs`: turnover and transaction-cost calculations.
- `pipeline`: signal composition and strict public backtest orchestration.
- `signal`: signal-date selection and execution scheduling.
- `portfolio`: `ScheduledPrediction` to weights policies.
- `allocation`: generic deterministic target weights to integer-lot positions.
- `engine`: package-private weight simulation.
- `factor`: information coefficient, quantile, and top-N signal evaluation.
- `performance`: summary metrics.
- `results`: dataclasses for downstream inspection.
- `visualization`: plotting helpers.

## Data Boundary

`AlphaValue` and weights are ordinary core `Panel` values. Composed predictions
are `PredictionPanel`; scheduling adds explicit date lineage in `ScheduledPrediction`.
Price frames contain numeric prices used to compute returns.

The package depends on `bagelquant-core` for Panel and composer contracts. It
does not import `bagelquant-data` or investment-domain application code.
