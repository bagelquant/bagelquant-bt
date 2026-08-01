# bagelquant-bt Documentation

`bagelquant-bt` composes AlphaValue panels into signals and evaluates those
signals. It does not retrieve market data.

The expected workflow is:

```text
AlphaValue Panel -> SignalComposer -> SignalPanel -> policies -> result
```

The package is Polars-first. Public backtests require core `SignalPanel`; prices
remain long-form Polars frames keyed by `time` and `asset_id`.

## Main Entry Points

```python
from bagelquant_bt import compose_signal, run_signal_backtest
```

Use `compose_signal` to create a typed signal and `run_signal_backtest` to apply
date, execution, market, and portfolio policies.

## Docs

- [Concepts](concepts.md)
- [Architecture](architecture.md)
- [Quick start](quick-start.md)
- [Performance notes](performance.md)
- [API](reference/api.md)
- [Public API](reference/public-api.md)
- [Transaction costs](reference/transaction-costs.md)
- [Signal evaluation](reference/factor-evaluation.md)
- [Internals](reference/internals.md)
