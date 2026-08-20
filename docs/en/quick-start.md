# Quick Start

`bagelquant-bt` composes AlphaValue panels into a strongly typed signal and
backtests only through that signal contract.

```python
from bagelquant_core import Domain, IdentityPredictionComposer, Panel
from bagelquant_bt import (
    BacktestConfig,
    EvaluationAnchor,
    AlphaPolicy,
    MissingSnapshotAction,
    compose_prediction,
    run_prediction_backtest,
)

alpha_value = Panel.from_domain(alpha_frame, domain, name="quality")
policy = AlphaPolicy(
    id="month_end",
    frequency="monthly",
    anchor=EvaluationAnchor.LAST_TRADING_DAY,
    missing_snapshot=MissingSnapshotAction.PREVIOUS_IN_PERIOD,
)
signal = compose_prediction(
    {"quality": alpha_value},
    IdentityPredictionComposer(),
    calendar,
    policy,
    standardization="zscore",
)
result = run_prediction_backtest(
    signal,
    prices,
    calendar,
    policy,
    config=BacktestConfig(initial_capital=1_000_000, top_n=50),
)
```

For `ICWeightedPredictionComposer`, `ICWeightedDecayPredictionComposer`,
`OLSPredictionComposer`, or `GLSPredictionComposer`, also pass `prices` to
`compose_prediction`. Their rolling window and half-life count signal
periods, not daily rows. Ordinary panels, raw DataFrames, and direct weights
cannot be passed to `run_prediction_backtest`.
