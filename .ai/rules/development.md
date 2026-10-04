# BT development rules

## Staged refactor target

The sequence is AI workflow, Data, Core, BT, Workbench, then a new database and
service restart. Step 1 records the design only; it changes no code, public API,
schema, numerical behavior, artifacts, database or running service. Later stages
may redesign incompatible APIs and remove obsolete code without compatibility
shims; real data, frozen evidence and operational actions retain their own
authorization boundaries. Concrete replacement APIs belong to their code stages.

BT owns scheduling, accounts/backtests, evaluation/statistics and the public
APIs for persisting, querying and reusing its account/evaluation artifacts and
checkpoints. Core owns generic numerical/training/optimization mechanics and
graph/numerical artifacts. Data owns neutral dataset, version, PIT and frozen-input
APIs. Workbench retains China declarations, authored definitions, app metadata,
governance, tasks/orchestration and Web/GUI; its metadata references backend
receipts. Target Workbench neither implements reusable engines nor accesses
backend private APIs, artifact files or backend database tables directly.

The dependency graph remains Data/Core independent, BT depending only on Core,
and Workbench consuming all three. Market rules remain explicit caller inputs;
BT never imports application/provider state. Current saved-result persistence
and version bindings are partly in Workbench. The account/evaluation topic rules
preserve current behavior until stages 4 and 5 transfer those mechanics.

## Preparation and boundaries

- Inspect `git status --short --branch` including untracked changes before editing;
  read README, manifest and affected docs. Work in the owning repository and preserve
  unrelated user edits. Never remove Git metadata, merge histories or absorb repos.
- Use Python 3.13 and `uv`. Choose cross-platform paths/APIs; account for separators,
  casing, line endings, permissions, shell syntax and environment conventions on
  Windows/macOS. Use `pathlib`; never persist developer-specific absolute paths.
- BT depends on Core, never Data or Workbench. Keep generic numerical primitives
  in Core, account/evaluation mathematics and artifact APIs here, and application/
  China market declarations downstream. Declare dependencies in the owner manifest.

## Implementation

- Prefer the smallest complete system and existing lower-level primitives. Keep
  one authoritative API/model/state/execution path per concern. Delete dead code
  rather than add adapters, aliases, flags or parallel implementations.
- Do not keep deprecated readers/routes/DSL aliases or migration shims unless
  an explicit compatibility window is requested. Add abstractions, dependencies,
  configuration or persisted state only for a current concrete responsibility.
- Keep dependencies and boundaries visible. Remove a feature's exports, tests,
  docs, configuration and generated references together; preserve real shared
  data, recovery evidence, user work, credentials, environments and caches.
- Use deterministic composable logic, descriptive names and typed public
  boundaries; follow surrounding structured docstrings. Validate boundaries
  with actionable errors and never silently swallow exceptions.
- Use `logging`, not `print`, in library logic. Prefer vectorized Polars/NumPy;
  row-wise loops need justification and measurement. Ordering/grouping are explicit.
- Performance changes preserve contracts unless explicitly changed/documented.
  Formula/metric/execution changes need hand-checkable examples and regression
  tests; compare optimized output with simple references where practical.
- Cover temporal alignment, costs, turnover, sparse calendars, listings,
  delistings, suspensions, gaps, missing/duplicate keys, empty/single-asset
  frames and non-trading dates where applicable. Use synthetic inputs and temporary
  roots; never read/mutate the real workspace data root or consume provider quota.

## Verification and delivery

- Run `uv run ruff check .` and `uv run pytest` for code changes; choose focused
  checks for docs-only work. Generate outputs from their source/generator;
  never manually edit generated docs, schemas or API artifacts.
- State cross-repo public contract changes. Keep edits independently coherent,
  update bounds/versions only when required, test the owner then every affected
  downstream consumer; report each repository separately.
- Major architecture changes update applicable AGENTS and owner rules in the
  same change; update root instructions if cross-repository boundaries change.
- Do not commit caches, credentials, databases, provider data, environments,
  build outputs, research artifacts or local task records. Use stable `main`
  and short-lived single-purpose branches; honor session branch instructions.
  Conventional Commit summaries are imperative and at most 72 characters.
- Commit/push/PR/merge/release/publication/deployment/service installation,
  real-data updates and governance transitions each require explicit request.
  Commit components separately before explicitly authorized gitlink updates.
  Package publication verifies version/build/registry/credentials/version absence;
  report exact uploaded versions/artifacts.
- Report behavior/why, repository-specific checks/results, unrun checks and reasons,
  and contract/version/data/operational caveats. Distinguish existing failures from
  introduced failures; never claim a check passed unless run.
