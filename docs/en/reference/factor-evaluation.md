# Alpha and Prediction evaluation

`evaluate_alpha` evaluates saved numeric/prediction Nodes using caller-aligned
one-session forward returns. Components select horizons, rolling IC, Book/Spread,
turnover, lead-lag, persistence, lagged alpha returns and quantiles. Direct IC,
rank IC and their IR plus persistence do not require an account.

Book uses centered cross-sectional average ranks with gross one. Spread assigns
+0.5/-0.5 to deterministic upper/lower quantile tails. `build_alpha_weights`
provides both constructions as weights Nodes; their normalized research NAV is
computed through the common weight path with proportional full-L1 costs.

Default cumulative windows are 1/5/10/20/40/60/120 sessions with corresponding
bucket diagnostics. Economic intervals/maturity remain explicit. Unmatured labels
are unavailable. Mature constituent gaps retain fixed Book/Spread weights with
visible expected/observed coverage; IC and complete quantile samples remain
strict. No reselecting/renormalizing using future label availability.

Cadence uses latest causal whole snapshots, a supplied calendar and explicit
anchor. Stable asset ordering resolves ties. Lag tests use caller-declared
calendar shifts; diagnostic negative lags are labelled research diagnostics,
never an executable plan. Shared functions compute rolling IC, persistence,
HAC/BH inference and risk/comparisons; applications do not copy those formulas.

Explicit `options: BTExecutionOptions | None` and `progress(completed, total)`
allow independent horizon windows to use caller-bounded parallel execution.
Defaults stay serial; outputs retain window order, numerical formulas and
availability. `EvaluationResult.execution` exposes worker/memory estimates
separately from tables and metrics. See [resources](../performance.md).
