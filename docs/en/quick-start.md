# Quick Start

`bagelquant-bt` composes AlphaValue panels into a strongly typed signal and
backtests only through that signal contract.

```python
from bagelquant_core import Domain, IdentitySignalComposer, Panel
from bagelquant_bt import (
    BacktestConfig,
    SignalAnchor,
    SignalDatePolicy,
    MissingSnapshotAction,
    compose_signal,
    run_signal_backtest,
)

alpha_value = Panel.from_domain(alpha_frame, domain, name="quality")
policy = SignalDatePolicy(
    id="month_end",
    frequency="monthly",
    anchor=SignalAnchor.LAST_TRADING_DAY,
    missing_snapshot=MissingSnapshotAction.PREVIOUS_IN_PERIOD,
)
signal = compose_signal(
    {"quality": alpha_value},
    IdentitySignalComposer(),
    calendar,
    policy,
    standardization="zscore",
)
result = run_signal_backtest(
    signal,
    prices,
    calendar,
    policy,
    config=BacktestConfig(initial_capital=1_000_000, top_n=50),
)
```

For `ICWeightedSignalComposer`, `OLSSignalComposer`, or `GLSSignalComposer`,
also pass `prices` to `compose_signal`. Their rolling window counts signal
periods, not daily rows. Ordinary panels, raw DataFrames, and direct weights
cannot be passed to `run_signal_backtest`.
