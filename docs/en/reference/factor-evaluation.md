# Alpha and Prediction evaluation

`evaluate_alpha` evaluates saved numeric/prediction Nodes using caller-aligned
one-session forward returns. Components select horizons, rolling IC, Book/Spread,
turnover, lead-lag, persistence, lagged alpha returns and quantiles. Direct IC,
rank IC and their IR plus persistence do not require an account.

`alpha_update_plan(previous, calendar=..., available_date=...,
affected_intervals=[(start, end)], preceding=[saved_state, ...])` inspects compact
state without value or table reads. It returns `initialize`, `up_to_date`, `append`,
`repair` or `blocked`, with `read_start`, `output_start` and `previous_index`:
`-1` selects the current result, a nonnegative index selects `preceding`, and
`None` selects no completed result. Repair includes horizon/persistence/lag impact
and restores the latest safe preceding receipt with the same policy/components.
Optional `input_proof` checks registered original-cutoff prefixes during planning.
Execute the admitted work through `evaluate_alpha(previous=selected_result)`;
missing state or unknown/all-history impact produces an explicit recovery plan.
Saved state includes the earliest pending horizon date, keeping arbitrarily
delayed labels in scope independently of the normal finite history window.

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

Pass `previous: EvaluationResult` and `input_proof(first, observation_through,
information_cutoff)` for unchanged-policy continuation. The proof binds typed
Alpha values, Domain/traces, calendar mappings and every caller label consumed
at that information boundary, including economic intervals and availability.
Observation and information boundaries are independent. A later information
cutoff can consume labels beyond the saved Alpha boundary. Unknown or changed
prefixes, regressing cutoffs, and missing primitive/checkpoint families require
explicit maintenance or affected-interval repair; they do not silently replay
completed history.

The result's `state` retains policy, boundaries, input prefix and independent
Book/Tail/lag/quantile weight checkpoints. Mature daily horizon primitives are
reused; pending maturity tails and new persistence targets continue. Each path
resumes from its actual checkpoint, including lag halos. `execution.path_sessions`
counts actual newly simulated path sessions. Summary/inference calculations still
consume saved primitives. Full typed input preparation and rolling-summary work
remain performance gaps; a small output tail alone does not prove bounded input I/O.

`BTStore.publish_continuation(tables, previous=references)` preserves equal
historical prefixes and writes changed daily tails. It currently consumes old
typed BT tables to identify those prefixes, including on unchanged-result reuse;
it avoids old-file hashing/writing but does not promise zero historical reads.
`publish_table` and `compose_reference` use the same immutable table authority.
Compositions select registered originals using indexed Date/String/Integer
coordinates, preserving typed order and exact selected row counts. Their logical
hash identifies the immutable composition rather than a flat table hash.
Publish a composition through `file_references` without a supplied frame.
Missing indexes require explicit `index_plan/build_index`; full verification
also checks coordinate facts against original rows. Publication checks original
file existence under the writer fence shared with cleanup.

`BTStore.maintain_shared_inputs` explicitly rebinds retained legacy signal
primitives when all consumed value, Domain, price, calendar and label projections
are proven identical at the original observation/maturity boundaries. Each
`SharedInputEvidence` declares the actual source cutoff, complete typed content,
selector, schema and traces; a source-owned report/reference and SHA256 retain
the attestation. BT validates declaration equality and the registered original
chain, while external source semantics remain the caller/source owner's duty.
No numerical files are decoded, recalculated or replaced. The same numerical
settings and original identity must reproduce exactly. Missing signal coverage
can be initialized separately once; groups, native scope and continuation state
cannot be added through this admission. Changed/unknown content or a later
evaluation boundary still requires an affected-interval plan or explicit causal
state maintenance; this API does not authorize historical replay.
