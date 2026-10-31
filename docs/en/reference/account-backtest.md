# Signed-share execution and FIFO

`evaluate_execution` consumes unique stable plan IDs and explicit dated signed
share quantities plus initial capital and caller prices/rules. Sells execute
first, then buys, with stable asset/plan ordering; rows are not netted.
Actual fills obey cash including fees, inventory/settlement, lots and blocks.
Remainders expire by default; explicit retries preserve only residual quantities.

The zero-fee reference fully executes original frozen quantities at supplied
reference prices, ignoring actual lot/settlement/blocking constraints. It must
remain cash/inventory feasible; reject the evaluation otherwise. Actual and
reference account/FIFO state are independent. Entry and exit fees allocate by
matched shares; remaining fees and unrealized P&L remain explicit. Cash dividends
belong to record-date originating plans. Bonus shares retain originating plans
with zero acquisition basis; this is generic P&L, not a tax-lot convention.

Optional `reference_price` on each plan overrides the market row's
`reference_price`; otherwise the reference uses the supplied execution price.
These are caller declarations. Reference prices and their selection participate
in checkpoint identity. A plan with no filled notional has a null return and a
typed `target_return_reason` or `actual_return_reason` of `no_filled_notional`.

`run_execution_from_weights` sizes complete saved weights at the actual decision
close and freezes signed deltas for caller-selected future execution. Opening
gaps never re-size shares. Known calendar must cover execution and settlement.
Decision sizing and actual fills both use `config.default_buy_lot_size` for
assets absent from the optional per-asset `lot_sizes` overrides.
For weight-mode execution with a non-unit default lot, public
`WEIGHT_SIZING_POLICY` (`bt.execution.weight_sizing.v2`) binds the corrected
sizing into source authority and standalone checkpoint proofs. A checkpoint
created under the former sizing implementation requires an explicit fresh
calculation; retained results are unchanged. Unit-default weights and explicit
transaction plans retain their existing identities.
Price gaps freeze last observed marks; recovery recognizes the cumulative move.
Its `reference_price_mode="execution"` default uses the supplied execution
reference, while explicit `"decision"` freezes the decision-close price alongside
the frozen quantity. Workbench selects `"decision"` for its target comparison.

Applicable corporate-action record/ex dates and cash/share payment dates within
the observed calendar range must be supplied trading sessions. Missing dates
raise an input error; BT does not choose a next-session convention or silently
skip events. Cash payment dates apply only to positive cash dividends and share
availability dates only to positive stock dividends. Future dates remain valid
before the calendar is extended on continuation, and verified old record dates
retain their existing history requirements.

Checkpoint/codec preserve both states, zero coordinates, FIFO entitlements and
future/retry plans. Resume verifies historical settings/input/calendar proofs;
newly known old-record actions need verified actual/reference positions, sessions
and open-lot histories. Cancellation is cooperative at session/order/planning
steps. `merge_execution_tables` appends verified history and replaces snapshots;
`summarize_transaction_pnl(..., through=end)` uses that cutoff's saved lot/mark
snapshot and mature events, never future fills or final holdings. No read replay.

For each plan, target/actual return equals the corresponding attributed P&L divided by that plan's filled notional; the money P&L and fee columns remain separately inspectable. Account returns use equity/NAV.

Native incremental callers supply `input_evidence` for the complete current
consumed range and `prefix_evidence` for each original checkpoint channel scope.
Both use `SharedInputEvidence`, including complete empty channels: weights and
decisions (or transactions), prices, calendar, availability, lot sizes, actions
and action coverage. `validate_execution_continuation(checkpoint_metadata, ...)`
checks these records before any numerical read. Source receipts may advance;
original content, schema, selectors, coordinates and traces must remain equal.
The `bt.execution.inputs.v1` binding separately records financial settings,
capital, lag and reference mode. Annualization stays report-only. Legacy signal
maintenance does not create this account authority; unknown old account evidence
requires explicit original-input checkpoint maintenance.
When a selector includes `asset_starts`, continuation preserves every original
asset's first consumed date. A newly admitted asset starts after the checkpoint;
its earlier unconsumed prices do not become historical account inputs.

Proven continuation supplies strictly new numerical rows plus the complete
calendar covering frozen execution/settlement dates. Checkpoints keep marks,
pending plans, future actions and cumulative per-plan attribution. Newly known
old-record actions require exact original position/lot/session dates; an effect
inside completed history requires a preceding checkpoint and suffix repair.
`result.tables["transaction_pnl"]` is cumulative without reading old fills or
matches. `result.tables["turnover"]` uses only new fills/values and preceding
checkpoint equity through `account_fill_turnover`.

Save tails through `BTStore.execution_references(frames, previous=..., through=...)`
and `publish_shared(file_references=..., expected_shared=...)`. Daily/event tables
retain their original prefix references, while lot/P&L/checkpoint snapshots
publish fresh small frames. The same boundary selection supports suffix repair.
`read_reference(reference, filters=[{"column": "time", "operation": "eq",
"value": cutoff}])` reads selected registered rows with leaf pruning and Parquet
predicate pushdown; indexed asset/ledger dates may be combined. Missing legacy
coordinate authority requires explicit `index_plan/build_index` maintenance.
Native indexes preserve original column order and registered coordinate types,
including sparse and all-null date keys. `verify_reference(deep=True)` compares
the index against the original table; `maintain_reference` explicitly registers
missing original facts without changing historical receipts or bytes.
