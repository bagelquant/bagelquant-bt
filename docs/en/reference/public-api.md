# Public API — BT 0.12

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

`evaluation_sample` and `SAMPLE_POLICY` expose the saved effective-sample contract; [research statistics](research-statistics.md) defines dates, gaps and empirical ES95.
