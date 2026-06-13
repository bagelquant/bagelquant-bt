# bagelquant-bt Documentation

`bagelquant-bt` measures research outputs. It does not build signals and it does
not retrieve market data.

The expected workflow is:

```text
daily data -> long-form factor or weights DataFrame -> bagelquant-bt result
```

The package is Polars-first. Prices and signal values must be long-form
`polars.DataFrame` objects keyed by `time` and `asset_id`.

## Main Entry Points

```python
from bagelquant_bt import BacktestConfig, run_backtest

result = run_backtest(
    weights,
    prices,
    kind="weights",
    config=BacktestConfig(initial_capital=1_000_000),
)
```

Use `kind="weights"` when the first argument is portfolio weights.

Use `kind="factor"` when the first argument is factor scores.

## Docs

- [Concepts](concepts.md)
- [Architecture](architecture.md)
- [Quick start](quick-start.md)
- [API](reference/api.md)
- [Public API](reference/public-api.md)
- [Transaction costs](reference/transaction-costs.md)
- [Factor evaluation](reference/factor-evaluation.md)
- [Internals](reference/internals.md)
