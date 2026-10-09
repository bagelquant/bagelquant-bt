# Architecture and artifact authority

Portfolio mechanics construct weights, select caller-declared schedules, drift
positions, freeze shares and simulate accounts. Pure evaluation functions consume
those results for IC, returns, risk, inference and comparisons. The signed-share
engine is the only execution loop; account.py contains its private primitives.

Dependencies: Data and Core are independent; BT depends only on Core; Workbench
composes public Data/Core/BT APIs. BT never imports Data or Workbench or resolves
providers/application state. Callers supply neutral panels, calendars, explicit
rules and source receipt references. BT does not create or train upstream values.

`BTStore(meta_path, artifact_path)` owns SQLite metadata and immutable Parquet
artifacts. `begin` identifies a numerical evaluation; `publish_shared` and
`publish_chapter` publish complete artifacts and receipts. Public `read_reference`,
`read_chapter`, `history`, `inventory`, `invalidate`, `verify`, `recover`,
`cleanup_plan` and `cleanup` manage this authority. Publication failure preserves
valid existing receipts; incompatible databases require a fresh store rather
than startup migration. Consumers retain references, not backend bytes or proofs.

Execution checkpoints retain actual/reference state, FIFO/entitlement evidence
and frozen future plans. Canonical input prefixes guard continuation. Append uses
`merge_execution_tables`; earlier windows use saved cutoff snapshots via
`summarize_transaction_pnl`, never current final lots or account replay.

Workbench owns global admission, hardware/runtime policy, China-market
interpretation, authored definitions, governance and app/global-version metadata.
BT exposes explicit local limits and workers. No stage-4 source cleanup authorizes
changing live data/services or deleting historical/authored evidence.

## Receipt inspection and selective reads

BTStore owns immutable chapter receipts and table integrity. Ordinary chapter/selected
table reads use registered metadata, row counts and indexed schema checks (`verify=False`).
Explicit `verify` retains full byte/logical/index audits. Registered references can
be published without decoding; new numerical frames retain canonical content hashes.
`index_plan/build_index` is explicit historical descriptor maintenance. Opening a store
never performs backfill or changes old receipts. Read-context audit proofs expire on exit.
