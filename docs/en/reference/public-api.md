# Public API — BT 0.12

`plan_risk_profile_update(previous_state, *, calendar, through,
affected_intervals=(), preceding=(), input_proof=None, ...)` returns a
`RiskProfileUpdatePlan` with mode, read_start/read_end, output_start/output_end,
specific_start and previous_index. Unknown state/prefix yields blocked work.
`rolling_risk_profile(..., previous=selected, plan=plan, input_proof=...)`
returns delta returns/exposures/diagnostics, compact state and an identity-neutral
solve_count. The default 240-session regression needs only its input halo and
119 saved specific returns. State retains policy, calendar_start, through/cutoff,
output_origin, pending_output_start, proof_through, factors and input_prefix.
Factor names become visible causally under `RISK_PROFILE_POLICY`; unchanged ridge
and maturity formulas own both initialization and continuation. Fixed-population
repair uses a proved preceding state and a finite regression/volatility halo.
Supply `previous_input_proof` from the latest original input receipts to attest
the complete prefix before a repair, including its saved specific-return halo.
Simultaneous repair/append needs separate admitted plans.

`BTStore.reference_evidence(reference, filters=())` returns registered typed schema,
factor/family/first-seen catalogue and date-content tokens.
`reference_interval_identity(reference, start=..., end=..., cutoff=...)` combines
compact date/availability summaries independently of artifact fragmentation,
without reading values. Unknown old summaries return None; explicit
`index_plan(evidence=True, references=[...])`/`build_index` adds them by auditing
original bytes, preserving receipts and results. New publication captures them
while consuming the admitted frame; no row-hash arrays are stored.

Both reference methods accept `time_column="time"`; an explicit
`time_column="interval_start"` groups/clips saved Gross source rows by their
economic start while retaining the original signal label, interval end and
availability in the proof. `interval_start`/`interval_end` are indexed Date axes
for bounded `read_reference(filters=...)`. Missing original axis summaries return
None and need explicit `index_plan(evidence=True, time_column="interval_start",
references=[...])` maintenance. A caller maps the actual saved economic interval
to numerical time; BT does not invent an execution lag or shift source dates.

`BTStore.publish_continuation(delta_frames, previous=references,
output_start=..., output_end=...)` composes the original prefix, delta and optional
unaffected suffix without decoding history. Omit bounds for complete-frame
continuation, which locates the first exact ordered mismatch in one pass.
`read_context(cache_bytes=...)` reserves one quarter of its finite cache for
metadata blocks and the remainder for decoded numerical frames under an
explicit 64 MiB default cap; zero disables caching and exit/failure clears it.

`alpha_update_plan(previous, *, calendar, available_date, affected_intervals=(),
preceding=(), ..., input_proof=None)` plans metadata-only append/affected-suffix
work and selects a retained original checkpoint. It returns a blocked recovery
plan when unchanged historical work cannot be safely reused; execution continues
through `evaluate_alpha(previous=selected)` and the existing weight engine.

`SharedInputEvidence` and `BTStore.maintain_shared_inputs(owner_id, ...)` provide
explicit same-boundary input maintenance for retained legacy signal primitives.
Supply original/replacement input identity bodies, the original receipt/request,
settings/kernel, scoped read windows, all five complete source-owned typed
projections and an immutable report reference/SHA256. BT publishes registered
references only and checks the original authority at reuse and publication.
This preserves groups/files and grants no append or causal continuation state;
see [Alpha evaluation](factor-evaluation.md) for scope and limitations.

Alpha continuation accepts `previous` and a three-boundary `input_proof`.
`EvaluationResult.state` retains path checkpoints and the prefix policy. See
[Alpha evaluation](factor-evaluation.md) for the proof contract and current
performance limits. `BTStore.publish_table`, `compose_reference` and
`publish_continuation` compose registered originals and changed daily tails
through the same immutable authority. Composed publication uses references only;
original-byte audits remain explicit.

Native execution also accepts `input_evidence` and `prefix_evidence` mappings of
complete `SharedInputEvidence` channels. `validate_execution_continuation`
checks original checkpoint metadata before numerical inputs are read. The
versioned account binding and cumulative summary are distinct from legacy signal
maintenance. `BTStore.execution_references` saves account tails and checkpoint
snapshots without decoding old daily tables; `read_reference(filters=...)`
prunes indexed leaves and pushes eq/ge/le coordinate predicates into Parquet.
`read_reference(coordinates=typed_keys)` selects exact Date/String/Integer key
tuples (including asset/available-date pairs); empty keys open no value files.
Native indexes also cover checkpoint_date and available_date. Missing coordinate
authority requires explicit `maintain_reference`, preserving original bytes.
See [execution](account-backtest.md) for required channels and failure modes.

- `evaluate_alpha(alpha, forward_returns, ...)` accepts a saved Core numeric or
  prediction Node. `evaluate_weights(weights, forward_returns, ...)` accepts a
  saved weights Node. Both return tables/metrics and the weight mode a checkpoint.
- Returns require time, asset_id, forward_return, interval_start, interval_end
  and available_date. Time is the caller's alignment key; economic intervals
  must be sequential/nonoverlapping. No implicit execution lag/price convention.
- `build_alpha_weights(alpha, method="book"|"spread", every=..., calendar=...,
  anchor=...)` builds gross-one/net-zero weights from latest causal snapshots.
- `evaluate_execution(transactions, market_prices, initial_capital=..., ...)`
  consumes stable plan_id, execution_date, asset_id and signed integer quantity.
  Prices explicitly provide execution_price/valuation_price or open/close.
  Optional per-plan reference_price overrides the market reference_price;
  otherwise the ideal account uses the supplied execution price.
- `run_execution_from_weights(weights, decisions, market_prices, calendar=...,
  session_lag=..., initial_capital=..., ...)` uses actual decision-close state
  to freeze deltas; rebalance/hold/unavailable stay explicit. Frozen shares are
  never resized at the execution price. Decisions have time/status columns.
  reference_price_mode="execution" is the default; "decision" explicitly freezes
  decision-close reference prices. Workbench chooses "decision".
- `save_checkpoint`, `load_checkpoint`, `merge_execution_tables` and
  `summarize_transaction_pnl(frames, through=...)` own continuation and saved
  historical account views. New record-date discoveries require verified actual
  and reference position/session/lot histories. No historical replay on reads.
- `ExecutionConfig` owns neutral money costs/lots/settlement/retry/annualization.
  `ExecutionCostRule` requires stable ID/version/JSON parameters and a pure valid
  monotone-cost quote returning `ExecutionCostQuote`.
- `return_statistics` and the public financial/risk/inference functions operate
  on explicit saved tables. BT defaults annualization 252 and quantiles 10;
  Workbench explicitly chooses 240 and China market rules.
- `BTStore`, `evaluation_identity`, `BTExecutionOptions` and
  `run_evaluation_batch` own result persistence and bounded local execution.

No public evaluation fetches providers, trains/builds upstream graphs, probes
hardware or changes governance. [Execution](account-backtest.md),
[costs](transaction-costs.md) and [architecture](../architecture.md) define details.

`BTStore(meta_path, artifact_path).inspect()` is the read-only setup check. It
returns `uninitialized`, `ready` or `incompatible` with version/reason information.
Readiness requires the current version, required tables and required columns.
It uses Core's public metadata snapshot primitive to include committed WAL
without opening the source in SQLite. Private temporary metadata copies are
removed after inspection; source directories, database bytes and WAL/SHM remain
unchanged. Active/hot rollback journals are refused without recovery;
invalidated zero-header PERSIST journals remain readable. It never recovers or
overwrites evidence.
Initialize empty stores only through explicit `initialize()`.

`evaluate_alpha` additionally accepts `options: BTExecutionOptions | None` and
`progress: Callable[[int, int | None], None] | None` for bounded horizon workers.
`EvaluationResult.execution` carries execution-only estimates independently of
numerical tables/metrics; never include it in evaluation identity.

`BTStore.inspect(runtime=True)` is the explicit active-service readiness mode.
It uses Core's coordinated read-only committed SQLite view without copying live
metadata, initializing schemas, recovering storage or accepting uncommitted
changes. Normal WAL read-mark coordination may touch SHM; durable payloads stay
unchanged. The default `runtime=False` retains strict offline inspection and
source/sidecar preservation for setup and installation.

`evaluation_sample` (including optional `minimum_signal_coverage`) and `SAMPLE_POLICY` expose the saved effective-sample contract; [research statistics](research-statistics.md) defines dates, gaps and empirical ES95.


`BTExecutionOptions.native_admission` optionally accepts
`Callable[[str], ContextManager[None]]`. BT calls it with `"mixed"` for complete
local evaluation work and keeps inner execution serial. This execution-only
admission is excluded from option equality and all numerical identity. Without
it, `workers` retains the standalone bounded-thread behavior.


`pair_correlation(left, right, *, kind, method="pearson", changed_dates,
previous=None, start=None, end=None, zero_missing=False, ...)` computes saved-input
return or daily cross-section correlation. Inputs are typed Date `time`, Float64
`value`, plus String `asset_id` for cross-sections; keys are unique. Return
correlation uses the common finite dates, default minimum 12, and Pearson only.
Cross-sections use common finite assets, default minimum three, Pearson or average
tie-ranked Spearman after matching, and an equal mean over valid dates. Constant
or insufficient samples are unavailable with a reason. `zero_missing=True` uses
the union of assets held by either account, excluding keys with zero weight in
both accounts and filling absent keys only; callers must first prove both daily
snapshots complete. Explicit nulls/nonfinite values never become zero.

The result includes compact daily `state`, `processed_dates`, correlation, sample
counts and coverage. A same-policy previous state retains unchanged dates; only
`changed_dates` are replaced or deleted and final range clipping uses summaries.
The caller owns source/access identity, batching, resource admission and memory
lifetime. BT never persists this transient state. Account unit-return and actual
holdings references now capture compact date evidence during publication. Older
ones return unknown until explicit original-bound `index_plan(evidence=True,
references=[...])` / `build_index` maintenance; passive reads never backfill.

Optional validated coordinate filters restrict daily evidence before hashing,
including maturity axes. `date_rows` reports exact selected counts for bounded
reader admission. Extending a cutoff preserves unchanged per-date tokens;
metadata reads never register a new composed reference or read numerical bodies.
