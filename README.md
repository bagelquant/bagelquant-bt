# bagelquant-bt

`bagelquant-bt` provides typed signal composition, evaluation, portfolio-policy
application, performance metrics, and Plotly visualization.

The public investment flow is:

```text
AlphaValue Panel -> AlphaPolicy -> StandardizePolicy -> PredictionComposer
-> PredictionPanel -> ExecutionPolicy -> WeightPolicy -> Weights Panel
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
    standardize_policy="z_score",
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
For prediction-only research, `run_prediction_horizon_diagnostics` decouples
evaluation cadence from economic horizon. Daily predictions are evaluated on
fixed cumulative `1/5/10/20/40/60/120D` windows and `1D`, `2–5D`, `6–20D`,
`21–60D`, and `61–120D` buckets. It publishes centered-rank gross-one Book
returns, gross-one Tail returns, full quantile curves, IC, signal persistence,
and Bartlett Newey–West inference without constructing a portfolio NAV.
For a structurally valid Book or Tail, a member whose forward label is missing
contributes zero for that window while its original weight is preserved. The
reported expected/observed counts and coverage ratio still expose the gap; the
cross-section is never reselected or renormalized. Quantile completeness and IC
rules remain strict and unchanged.
`run_daily_rank_path_diagnostics` complements those forward-window statistics
with a daily-rebalanced, capital-free research path. Gross is the requested
Book/Tail factor return; Net subtracts proportional commission, sell tax,
transfer fees, and slippage from requested weight changes. Initial capital,
cash, minimum fees, execution blocks, and insolvency state never enter these
returns. Execution availability affects only the separately reported executed
turnover. It also publishes ten continuous daily-rebalanced equal-weight
quantile Gross paths. Missing asset returns freeze at zero and recover later
without reselecting or renormalizing the group. The result marks the true
initial rebalance, evaluates the Book over a common-sample integer lead-lag grid from
`-30` through `+30` sessions, and publishes common-sample Book/Tail paths for
lags `0/1/2/5/10/20/60`.
Applications that need both result families should call
`run_daily_prediction_diagnostics`. It validates and collects the scheduled
Prediction once, prepares the market/calendar and rank weights once, streams
one asset-label window at a time, and reuses one `1..120` autocorrelation pass.
The two narrower entry points remain available for independent analysis.
`AlphaPolicy` owns only evaluation-date alignment. `StandardizePolicy` is the
independent cross-sectional preprocessing contract (`none`, `z_score`, or
`percentile_rank`). Prediction composition returns the composer's raw
`PredictionPanel`; callers may explicitly apply Core Transformers before the
Weight Policy boundary.

`PredictionRegularizedOptimizerPolicy` converts a Prediction plus explicit
reference weights into target weights under long-only, fully-invested, and
maximum-weight constraints. CVXPY is available through the `optimizer` extra;
OSQP is attempted first and CLARABEL second, with strict failure if neither
solver succeeds.

`allocate_integer_positions` is the public deterministic bridge from one
continuous target snapshot to whole-lot positions. It first maximizes deployed
stock notional under the stock budget, then minimizes absolute notional
deviation among maximum-deployment solutions with stable `asset_id`
tie-breaking. Callers provide prices, lot sizes, minimum frozen quantities, and
whether each asset may exceed its continuous target by one lot; infeasible
inputs fail explicitly.

The separate `run_account_backtest` engine sizes target weights into integer
positions and simulates cash, T+1 availability, lot rules, sell-first funding,
orders, fills, unadjusted open/close marks, corporate-action receivables,
external-flow units, pending withdrawals, and performance attribution. It
returns `AccountBacktestResult` and does not change the fractional-weight
engine used by research diagnostics. See
[Whole-share account backtests](docs/en/reference/account-backtest.md).

`run_planned_account_backtest` is the causal execution counterpart for a
decision system. It accepts quantities frozen from decision-date target
weights, notional, and prices, executes those quantities on the declared next
session without sizing them again at the open, and expires every unfilled
remainder at the end of that session. Its resulting executable weights are
actual account weights and remain distinct from the frozen target weights.

`run_stateful_account_backtest` combines decision-close sizing and later
execution in one causal account pass. Its callback sees the actual checkpoint
and reference weights at each decision close, returns target weights once, and
the engine freezes whole-lot quantities without replaying prior sessions.

During an asset-specific price gap, its holding is frozen at the last observed
price. A held asset without a decision-close price remains outside that
decision's immutable execution plan, while every newly targeted asset still
requires a finite close. The gap sessions have zero return and the cumulative
move is recognized when a new price appears.

## Development

```bash
uv run ruff check .
uv run python -m pytest
```
