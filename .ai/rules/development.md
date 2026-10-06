# BT development rules

## Current BT 0.12 stage-4 contracts

BT evaluates saved Core numeric/prediction Nodes (`evaluate_alpha`), saved
weights Nodes (`evaluate_weights`) and frozen signed share plans
(`evaluate_execution`). Portfolio construction/scheduling/account mechanics and
pure evaluation functions are separate layers. The weight-to-share bridge uses
actual decision-close state and the same account engine; execution never resizes
frozen quantities. Removed policy/composition/account runners have no aliases.

BTStore owns its SQLite metadata, immutable Parquet tables, numerical identity,
receipts, publication integrity, invalidation, recovery and cleanup APIs.
Workbench owns app/global-version bindings, governance and backend references;
it uses public APIs rather than BT tables/files or duplicate numerical caches.
Core owns graphs/numerical/model artifacts; Data owns source datasets/PIT/input
proofs. BT depends only on Core and never fetches providers or imports Data or
Workbench. Stage 5 may clean other Workbench concerns. Database/service cutover
is stage 6 and remains separately authorized.

## Engineering and validation

Inspect tracked/untracked Git state, README, pyproject and routed owner docs.
Preserve unrelated changes and each repository's Git metadata. Use Python 3.13,
uv, pathlib and explicit deterministic ordering. Keep one authoritative model,
formula, engine, execution path and artifact owner; remove dead exports/tests/
configuration/docs together, with no unrequested compatibility/migration layers.

Validate typed inputs at boundaries with actionable errors. Financial state is
causal; outputs use bounded columnar row buffers. Prefer vectorized numerical
kernels where causality permits. Formula changes require hand-checkable examples,
reference comparisons and sparse/missing/duplicate/empty input cases.

Local parallelism uses BTExecutionOptions and caller Core resource ceilings.
No CPU/RAM probing or automatic global allocation. Resource/batch/worker settings
are identity-neutral; numerical settings, cost rules and source receipts are not.
Cancellation stops admission and running causal account work cooperatively.

Run `uv run ruff check .` and `uv run pytest` in this owner, then affected
consumers. Full work receives independent review. Verify standalone APIs,
immutable publication, changed-prefix rejection, append/full equivalence,
corruption/recovery and cleanup. Tests use synthetic data and isolated temporary
stores; never touch the real workspace data root or call providers.

Use owning generators for generated references, catalogs, OpenAPI and types.
Architectural changes update entrances/rules/docs and distinguish implementation
from remaining gaps. Record per-owner actual verification and unrun checks.

No commit/push/PR/merge/release/deploy/service/provider call/real data update or
governance transition without explicit authorization. Never delete authored
research, immutable historical evidence, recovery material, credentials,
environments or ignored machine state as source cleanup. Formal task metadata
belongs to one workspace coordinator, with disjoint subagent scopes.
