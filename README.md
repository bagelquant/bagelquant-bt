# bagelquant-bt

`bagelquant-bt` provides typed signal composition, evaluation, portfolio-policy
application, performance metrics, and Plotly visualization.

The public investment flow is:

```text
AlphaValue Panel -> SignalComposer -> SignalPanel -> SignalDatePolicy
-> ScheduledSignal -> ExecutionPolicy -> PortfolioPolicy -> Weights Panel
```

Public backtests require a core `SignalPanel`. Ordinary panels, raw DataFrames,
and direct weights cannot enter the backtest API. Prices remain explicit
long-form Polars data with `time`, `asset_id`, and `price`.

```python
from bagelquant_core import IdentitySignalComposer, Panel
from bagelquant_bt import (
    BacktestConfig,
    MissingSnapshotAction,
    SignalAnchor,
    SignalDatePolicy,
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
)
result = run_signal_backtest(
    signal,
    prices,
    calendar,
    policy,
    config=BacktestConfig(initial_capital=1_000_000),
)
```

`run_signal_evaluation` computes execution-to-execution IC, quantiles, spread,
TOP N, lag, and IC-decay diagnostics from `ScheduledSignal`. Sparse or monthly
signals rebalance only on their snapshot dates; positions are held between
snapshots. `OrderSizingPolicy` reserves the later target-weight-to-order
boundary, but quantity sizing is not implemented yet.

During an asset-specific price gap, its holding is frozen at the last observed
price. The gap sessions have zero return and the cumulative move is recognized
when a new price appears.

## Development

```bash
uv run ruff check .
uv run python -m pytest
```
