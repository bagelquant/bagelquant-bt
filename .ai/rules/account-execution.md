# Typed contracts, targets and account execution

These are current API/numerical constraints. The
[staged target](development.md#staged-refactor-target) assigns reusable account
artifact storage and checkpoint persistence to BT; the present Workbench bindings
below remain baseline facts until the owning refactor stages implement that API.

## Boundaries and scheduling

- Depend on Core Panel/Prediction contracts; never import Data/Workbench or assume
  China/application-specific behavior in generic APIs. BT owns simulation and
  evaluation, not upstream value production or training.
- `run_prediction_backtest` requires `PredictionPanel`, never a plain Panel,
  raw frame or direct weights. `ExecutionPolicy.schedule_prediction` produces
  `ScheduledPrediction` for prediction diagnostics/evaluation. The separate
  saved-target account APIs accept explicit target frames; in particular,
  `evaluate_portfolio_targets` consumes complete saved targets plus
  rebalance/hold/unavailable decisions. Do not conflate these input contracts.
- `compose_prediction` and `compose_processed_prediction` return the Composer's
  raw typed `PredictionPanel`. Do not add fixed/implicit normalization;
  callers explicitly express post-composer transformations.
- Prices use `(time, asset_id, price)`; Prediction/weights use Core's
  `(time, asset_id, value)`. Align snapshots to exact observed price keys.
  Executed weights at `time=t` earn the next market-session close-to-close return.
- Keep signal selection, execution schedule, portfolio policy, returns, costs,
  metrics, results, reports and figures separable. Sparse/monthly signals only
  rebalance on snapshot dates and hold between them. NaN is never an order;
  full target zeros exit assets, hold/unavailable states stay explicit.
- Distinguish observation, publication/information cutoff, signal-effective and
  execution dates; no latest-value substitution or future information at time t.
- `PredictionRegularizedTargetVolatilityPolicy` is a separate Weight Policy;
  do not change `PredictionRegularizedOptimizerPolicy` to implement it. First
  construct the unchanged fully invested risky sleeve, estimate volatility from
  strictly prior independent gross sleeve returns, skip incomplete warm-up,
  scale exposure without renormalization and retain residual zero-return cash.

## Account mechanics and continuation

- BT alone simulates fills/lots/T+1/cash/costs/suspension/limits/corporate actions
  from caller-supplied market rules. Research backtests, whole-share simulated
  accounts and orders/fills are in scope; broker connectivity, production order
  planning, real-account state and live submission are not.
- During an asset price gap, freeze existing holdings at the last observed
  price, recognize the cumulative move on recovery, do not silently resize/open
  blocked assets and leave blocked new target weight as cash.
- Turnover and costs reflect actual target changes. Keep commission/minimum
  fees/slippage/sell tax/insolvency explicit and reproducible. Rank and allocation
  ties use stable `asset_id`; optimizers reference computed targets, not accounts.
- Checkpoint continuation retains zero-quantity coordinates, original target
  dates and frozen future decision-close plans. Never rerun old callbacks,
  resize frozen quantities at execution open, or reprocess checkpoint decisions.
- Saved-target schedule kernel v2 requires known future trading dates and
  freezes their calendar proof; prices still stop at Available Date. Persist
  unexecuted plans with the account checkpoint and restore them on continuation.
  Schedule changes affect account evaluation identity, not saved Portfolio targets.
- Newly mature actions with pre-checkpoint record dates require verified saved
  positions and complete prior account-session inventory; never use current
  shares for historical entitlement. Prefix proofs use canonical row order,
  never physical scan order. Missing required prior history fails explicitly.
- `PreparedAccountMarketData` shares validated prices and execution blocks only;
  independent accounts/scenarios retain separate cash, positions, fills and state.
  Consumers prepare it lazily only for accounts that need computation.
- Keep frozen evaluation kernel versions; unavailable historical kernels require
  a new forward validation batch, never rewritten immutable registrations/evidence.
  Currently Workbench holds account-kernel and Portfolio decision/checkpoint
  version bindings; target BT owns account artifact/checkpoint mechanics while
  Workbench retains application bindings to the backend receipts.
- Flush account output rows in bounded columnar batches; preserve complete
  public schemas and causal date order. Share prepared inputs under the total
  resource budget; operational limits never change numerical identity.
- Importing BT must not initialize SciPy optimization/statistics. Load those at
  their numerical call sites while retaining public exports and behavior.
