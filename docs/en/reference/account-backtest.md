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
Price gaps freeze last observed marks; recovery recognizes the cumulative move.
Its `reference_price_mode="execution"` default uses the supplied execution
reference, while explicit `"decision"` freezes the decision-close price alongside
the frozen quantity. Workbench selects `"decision"` for its target comparison.

Checkpoint/codec preserve both states, zero coordinates, FIFO entitlements and
future/retry plans. Resume verifies historical settings/input/calendar proofs;
newly known old-record actions need verified actual/reference positions, sessions
and open-lot histories. Cancellation is cooperative at session/order/planning
steps. `merge_execution_tables` appends verified history and replaces snapshots;
`summarize_transaction_pnl(..., through=end)` uses that cutoff's saved lot/mark
snapshot and mature events, never future fills or final holdings. No read replay.

For each plan, target/actual return equals the corresponding attributed P&L divided by that plan's filled notional; the money P&L and fee columns remain separately inspectable. Account returns use equity/NAV.
