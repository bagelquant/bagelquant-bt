# bagelquant-bt

`bagelquant-bt` provides typed signal composition, evaluation, portfolio-policy
application, performance metrics, and Plotly visualization.

The public investment flow is:

```text
AlphaValue Panel -> AlphaPolicy -> PredictionComposer -> PredictionPanel
-> ExecutionPolicy -> WeightPolicy -> Weights Panel
```

Public backtests require a core `PredictionPanel`. Ordinary panels, raw DataFrames,
and direct weights cannot enter the backtest API. Prices remain explicit
long-form Polars data with `time`, `asset_id`, and `price`.

```python
from bagelquant_core import IdentityPredictionComposer, Panel
from bagelquant_bt import (
    BacktestConfig,
    MissingSnapshotAction,
    EvaluationAnchor,
    AlphaPolicy,
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
)
result = run_prediction_backtest(
    signal,
    prices,
    calendar,
    policy,
    config=BacktestConfig(initial_capital=1_000_000),
)
```

`run_prediction_evaluation` computes execution-to-execution IC, quantiles,
spread, TOP N, lag, and IC-decay diagnostics from `ScheduledPrediction`.
Prediction composition returns the composer's raw `PredictionPanel`; callers
may explicitly apply Core Transformers before the Weight Policy boundary.

`PredictionRegularizedOptimizerPolicy` converts a Prediction plus explicit
reference weights into target weights under long-only, fully-invested, and
maximum-weight constraints. CVXPY is available through the `optimizer` extra;
OSQP is attempted first and CLARABEL second, with strict failure if neither
solver succeeds.

The separate `run_account_backtest` engine sizes target weights into integer
positions and simulates cash, T+1 availability, lot rules, sell-first funding,
orders, fills, unadjusted open/close marks, corporate-action receivables,
external-flow units, pending withdrawals, and performance attribution. It
returns `AccountBacktestResult` and does not change the fractional-weight
engine used by research diagnostics. See
[Whole-share account backtests](docs/en/reference/account-backtest.md).

During an asset-specific price gap, its holding is frozen at the last observed
price. The gap sessions have zero return and the cumulative move is recognized
when a new price appears.

## Development

```bash
uv run ruff check .
uv run python -m pytest
```
