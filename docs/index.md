# bagelquant-bt Documentation

`bagelquant-bt` measures research outputs. It does not build signals and it does
not retrieve market data.

The expected workflow is:

```text
daily data -> factor or weights DataFrame -> bagelquant-bt result
```

The package is DataFrame-first. Prices and signal values must be numeric
`pandas.DataFrame` objects.

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
- [API](api.md)
- [Transaction costs](transaction-costs.md)
- [Factor evaluation](factor-evaluation.md)
