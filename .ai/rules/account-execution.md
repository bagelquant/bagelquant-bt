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

Use public codec, merge_execution_tables and summarize_transaction_pnl. Historical
summaries use the cutoff's saved lot/mark snapshot and mature events, never the
latest open-lot/per-plan table. Period reads do not replay accounts or write caches.
Account loop cancellation checks each session/order/planning step. Internal
account.py contains private reachable primitives and no second execution loop.
