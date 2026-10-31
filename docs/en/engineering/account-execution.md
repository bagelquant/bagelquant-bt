# Portfolio and execution contracts

## Research weights

Book centers cross-sectional average ranks and normalizes full absolute weight
to one; Spread allocates +0.5 and -0.5 to deterministic top/bottom tails. Both
are net-zero. Rebalance cadence uses the latest finite causal whole snapshot,
anchored to a supplied calendar/start; NaN is never an order. Complete targets
include zero exits; hold/unavailable states remain explicit.

Research weights drift between rebalances. Cost equals rate times the full L1
change from the Net account's drifted pretrade weights, including initial entry;
default rate is 0.0005. Costs subtract directly from returns in virtual cash.
Gross and Net drift independently. `net_pre_cost_return - cost_return` equals
Net return; independent Gross return is not required to satisfy that equality.
Minimum money fees, fills, lots, cash constraints and capital do not enter this
research account.

## Execution

Plans carry unique stable plan_id, execution_date, asset_id and signed integer
quantity. No netting or default lag/price selection. Explicit market inputs use
execution_price/valuation_price (or caller-specified open/close fields). Sell-first
order is deterministic by asset/plan ID. Actual buys respect cash including fees
and lots; sells respect available inventory. Missing prices/blocks prevent fills.
Unfilled quantities expire by default; explicit retry preserves only residual
shares, charging per order per new execution date. No negative cash or shorts.

Default costs are max(0.0005 * executed notional, 5), with no tax/slippage, lot 1
and settlement 0. Custom pure cost rules need stable ID/version/parameters and
nonnegative valid quotes; total buy charge must increase with quantity.
Caller supplies exchange-specific lots/settlement/tax/slippage/actions.
The weight bridge uses that same configured default lot during decision sizing;
per-asset overrides need not enumerate all assets.
For weight mode with a non-unit default, `WEIGHT_SIZING_POLICY` binds the fixed
sizing to native financial settings and standalone prefix proofs. Marker-free
old account state cannot continue under the corrected implementation. Explicit
transactions and unit-default weights preserve their existing proofs.
Applicable corporate-action dates inside the supplied observation range must
belong to its calendar.
Reject missing record/ex dates or positive cash/share settlement dates before
executing, rather than silently losing income or stranding receivables. Dates
after the cutoff may await a future calendar extension; original historical
entitlement proofs remain required on continuation.

The ideal zero-fee account fully executes original quantities at supplied
reference prices without actual blocks/lots/settlement constraints. It must
remain cash- and inventory-feasible; reject the entire evaluation otherwise.
The ideal and actual ledgers have separate FIFO basis, holdings and P&L.
Plan reference_price overrides market reference_price, otherwise the supplied
execution price is used. The bridge explicitly selects execution (default) or
decision reference mode; decision mode freezes decision-close prices. Reference
choices enter causal identity. Null per-plan returns carry typed reason columns.
Entry/exit fees allocate by matched quantities. Bonus shares have explicit zero
acquisition basis and retain originating plan IDs; cash dividends belong to
record-date lots. Realized/unrealized plus income reconcile equity every session.
These are investment P&L records, not an implicit tax-basis policy.

The saved-weight bridge sizes at actual decision-close state and freezes signed
deltas. No execution-date sizing or historical callback replay. Price gaps use
last observed marks and recognize recovery moves; never mark with future prices.
Known future calendar must cover execution and buy settlement.

## Continuation and views

Checkpoint retains both ledgers, active FIFO lots, per-lot entitlements, frozen
future/retry plans, zero asset coordinates and cumulative money totals. Resume
verifies canonical historical/settings/calendar prefix proofs before continuing.
Newly known actions with old record dates require verified actual/ideal position,
lot and complete session history; never substitute current holdings.

Native source-bound continuation uses `input_evidence`/`prefix_evidence` on the
same public engine, with `SharedInputEvidence` for every consumed channel,
including explicit empty inputs. `validate_execution_continuation` accepts
checkpoint metadata and runs before numerical reads. Source receipt changes are
neutral only when original complete scope/schema/selectors/traces/content tokens
match; financial settings, capital, lag and reference selection remain bound.
Annualization alone is not financial account rebuilding authority. The separate
`bt.execution.inputs.v1` checkpoint binding grants no legacy signal-maintenance
authority. Unknown legacy account source binding/cumulative state requires
explicit original-input maintenance; never invent it from current inputs.

After admission, supply strictly new prices/targets/decisions/availability/coverage,
new frozen plans, the complete proven calendar and action delta. Saved marks,
pending plans, settlement and active future actions continue from checkpoint.
New old-record actions read only their exact original positions/lot/session
dates; past effective effects require repair from a preceding checkpoint.
The native cumulative per-plan summary retains filled notional/cost/realized
income and recomputes current unrealized attribution from active lots. Daily
turnover uses preceding checkpoint equity and new fills through the same formula.

`BTStore.execution_references` retains indexed daily/event prefixes through the
restored boundary, appends only new tail tables and replaces small snapshots and
checkpoint frames. Earlier checkpoints repair a suffix while preserving prior
receipts/files. `read_reference(filters=...)` uses registered coordinate counts,
skips disjoint concat leaves and pushes predicates into Parquet. Missing indexes
require explicit `index_plan/build_index`, without implicit historical reads.
Exact multi-column `coordinates` use registered typed nonnull keys, metadata
pruning and Parquet axis predicates followed by an exact semi join; empty keys
open no numerical files. Native date indexes include checkpoint_date and
available_date. Explicit maintain_reference audits/indexes an original table
and leaves without modifying its receipt or bytes. Static consumed lot rules
remain checkpoint state; changing one used asset's rule blocks continuation.
An `asset_starts` selector retains each asset's original first consumed date;
new assets begin strictly after the restored checkpoint and never widen old
price/availability coordinates. Deep reference audits compare indexed column
order against the original table. Sparse coordinate metadata uses registered
types, including all-null date axes, without sample-based type inference.

Use public codec, merge_execution_tables and summarize_transaction_pnl. Historical
summaries use the cutoff's saved lot/mark snapshot and mature events, never the
latest open-lot/per-plan table. Period reads do not replay accounts or write caches.
Account loop cancellation checks each session/order/planning step. Internal
account.py contains private reachable primitives and no second execution loop.
