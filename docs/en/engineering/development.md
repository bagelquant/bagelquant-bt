# BT ownership, storage and execution contracts

## Current BT 0.12 stage-4 contracts

BT evaluates saved Core numeric/prediction Nodes (`evaluate_alpha`), saved
weights Nodes (`evaluate_weights`) and frozen signed share plans
(`evaluate_execution`). Portfolio construction/scheduling/account mechanics and
pure evaluation functions are separate layers. The weight-to-share bridge uses
actual decision-close state and the same account engine; execution never resizes
frozen quantities. Removed policy/composition/account runners have no aliases.

BTStore owns its SQLite metadata, immutable Parquet tables, numerical identity,
receipts, publication integrity, invalidation, recovery and cleanup APIs.
Schema 2 keeps compact headers in SQLite and exact bounded Zstd Parquet metadata
blocks under the BT artifact root. Ordinary startup rejects schema 1.
No schema converter or legacy index parser is provided. Incompatible stores
require explicitly authorized fresh initialization; never rewrite retained
receipt/reference bytes. See [metadata storage](../metadata-storage.md).
Workbench owns app/global-version bindings, governance and backend references;
it uses public APIs rather than BT tables/files or duplicate numerical caches.
Core owns graphs/numerical/model artifacts; Data owns source datasets/PIT/input
proofs. BT depends only on Core and never fetches providers or imports Data or
Workbench. Stage 5 implements thin Workbench composition. Stage 6 installs
services in Setup mode and initializes fresh stores only on explicit completion.
Live database/service operations remain subject to the current authorization.


BTStore.inspect is the public read-only setup probe; construction/inspection
create no storage or recovery state and reject mismatched versions, tables or
required columns unchanged. It reuses Core's public metadata snapshot primitive
for committed WAL inspection without creating or changing source sidecars.
Transient metadata copies are system-temporary, removed after inspection and
never contain BT numerical artifacts or become an additional authority.
Refuse active/hot rollback journals without recovery or reading uncommitted
schema pages; invalidated zero-header PERSIST journals remain readable.

`evaluate_alpha(options=..., progress=...)` uses the same bounded batch executor
for independent horizon windows. Shared prepared frames are read-only; each
worker owns its output. Conservative shared/workspace estimates reduce requested
workers, and small panels stay serial. `EvaluationResult.execution` reports
actual workers and estimates outside numerical metrics/identity. Callers select
outer target parallelism or inner window parallelism, never nested pools.

`BTStore.inspect(runtime=True)` is the explicit active-service readiness mode.
It uses Core's coordinated read-only committed SQLite view without copying live
metadata, initializing schemas, recovering storage or accepting uncommitted
changes. Normal WAL read-mark coordination may touch SHM; durable payloads stay
unchanged. The default `runtime=False` retains strict offline inspection and
source/sidecar preservation for setup and installation.

BTStore metadata APIs use query-only committed transactions; explicit initialization enables WAL.
Default chapter verification is metadata-only (`verify=False`). Ordinary selected table reads
check registered descriptors, row counts and indexed schemas without byte/logical rehashing.
Explicit `verify` retains full byte/logical/index checks and expiring context proofs.
`publish_shared/publish_chapter(file_references=...)` resolve registered original descriptors;
new frames retain canonical hashes, and unregistered conflicting files require canonical proof.
Optional table descriptor maintenance uses public `index_plan/build_index`; no implicit backfill.
Publication hashes frames once and resolves all descriptor misses in one streamed
original-receipt pass per batch. Literal-token rejection never supplies authority;
Unicode-escaped candidates still require exact nested files matches and receipt
authentication. Prepared misses recheck indexed authority and the append-only
receipt frontier; concurrent new receipts refresh remaining misses together.
Never infer new-table authority from index or file absence. Keep receipt rows
append-only (invalidation changes only status); cleanup never removes them.
Release owned raw/decoded/traversal aliases on failure without clearing foreign
exception frames. Scalar reads retain SQLite candidate filtering. New table
publication must not walk unrelated historical file trees to prove absence.
`read_context(cache_bytes=...)` splits its one finite cache budget between
metadata (one quarter) and numerical frames (the remainder), bounding both memos
(64 MiB default, zero disables it). It expires on exit/failure, rechecks source
file stats and never substitutes for registered evidence or explicit full audits.
Continuation compares ordered rows once when supplied full frames; admitted
output_start/output_end delta publication composes original prefix/suffix
references entirely from owner metadata, without scanning historical values.
Saved return `time` may be a caller's signal label. Public reference evidence
and interval identity accept an explicit `time_column` (default time); interval_start
and interval_end are indexed Date axes. Callers align numerical inputs from their
actual saved economic intervals, never a guessed date shift. Missing captured
economic-axis summaries require explicit original-reference index maintenance.


`BTExecutionOptions.native_admission` is an optional identity-neutral caller
context factory. Mixed evaluation units retain serial inner native work.
An explicit `window_dispatch` may admit a staged independent horizon batch:
the caller ends its parent native phase, drains child work and restores the
parent permit before BT reads/validates results. Idle successful workers may
remain for the operation; cancellation/failure reaps them before staging expires.
`HorizonWork` and
`compute_horizon_work` use the same window kernel as inline evaluation. BT
supplies complete workspace/retention estimates; caller policy selects fixed
CPU/memory shares before imports. No stores or publication run in children.
Standalone options retain bounded threads, ready-future refill and ordered
results. BT never imports the caller's admission implementation.

Local parallelism uses explicit BTExecutionOptions and caller Core resource ceilings; resource settings are identity-neutral. BT never probes CPU/RAM or allocates a global budget.
