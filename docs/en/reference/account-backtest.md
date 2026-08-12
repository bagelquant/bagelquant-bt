# Whole-share account backtests

`run_account_backtest` is an independent deterministic daily account engine.
It consumes target weights, provider-neutral unadjusted open/close prices,
corporate actions and coverage, execution availability, lot sizes, initial
cash/positions, and `AccountBacktestConfig`.

Each session restores the prior checkpoint, releases settlement and corporate
action receivables, marks at the open, applies fixed-notional flows when
configured, sizes integer target positions, executes eligible sells, pays a
pending withdrawal, and allocates affordable buy lots. Buy allocation chooses
the largest tracking-error reduction and breaks ties by stable `asset_id`.
Orders are deterministic full-quantity fills or explicit blocked/reduced
intent; the latest target revision supersedes earlier pending intent.

The engine never uses adjusted prices to synthesize shares. Record-date close
holdings establish dividend entitlements; ex-date creates cash and stock
receivables; pay-date releases cash; share-available date releases stock.
Coverage must be complete for every simulated market session.

Fixed-notional mode injects or requests removal of cash to maintain the chosen
notional. External flows change fund units, not unit NAV. A blocked withdrawal
remains explicit and the engine permits neither negative cash nor implicit
leverage. Compounding mode has no external flow and sizes from current equity.

`AccountBacktestResult` exposes target weights and positions, orders, fills,
daily positions, cash, receivables, external flows, pending withdrawals,
account equity, performance NAV, executable weights, target/implementation/cost
drag, and a resumable checkpoint.

