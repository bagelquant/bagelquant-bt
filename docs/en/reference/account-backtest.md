# Whole-share account backtests

`run_account_backtest` is an independent deterministic daily account engine.
It consumes target weights, provider-neutral unadjusted open/close prices,
corporate actions and coverage, execution availability, lot sizes, initial
cash/positions, and `AccountBacktestConfig`.

`run_planned_account_backtest` uses the same account mechanics but accepts an
immutable decision plan with `decision_date`, `execution_date`, `asset_id`,
`target_weight`, `sizing_notional`, `decision_price`, and `target_quantity`.
The execution open is used only for fills and affordability; it never resizes
the frozen target quantity. Any blocked or unaffordable remainder expires at
the end of that execution session and is not retried on a later date.

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

At a stateful decision close, an existing holding with no finite close remains
frozen at its last observed mark and is omitted from that decision's immutable
execution plan. A target asset still requires a finite decision-close price.
The holding can be resized by a later decision after its close price recovers.

When resuming `run_stateful_account_backtest`, pass previously frozen plans as
`initial_target_position_plans` alongside the checkpoint. A plan decided before
the checkpoint but due afterward executes on its original date and quantities,
without invoking its decision callback again or resizing at the execution open.
The saved-target `evaluate_portfolio_targets` entry point applies the same rule:
its calendar must include known future execution dates, while price coverage
stops at the simulation cutoff. Persist the returned future plans and supply
them on continuation. A truncated calendar that cannot schedule a decision is
rejected. `prepare_account_market_data` supplies validated prices and execution blocks to all
independent scenarios; it contains no account state.

Checkpoints retain zero-quantity coordinates and the original target revision
date, so later exits and pending-order audit identities match a continuous run.
If an appended corporate action has a record date before the checkpoint, pass
the verified prior `historical_positions` and `historical_sessions`. The engine
recovers entitlement from that record-date close, never current holdings. A
record date outside the complete prior account-session inventory has no account
entitlement. Missing history for a required prior session fails explicitly.

Fixed-notional mode injects or requests removal of cash to maintain the chosen
notional. External flows change fund units, not unit NAV. A blocked withdrawal
remains explicit and the engine permits neither negative cash nor implicit
leverage. Compounding mode has no external flow and sizes from current equity.

`AccountBacktestResult` exposes target weights and positions, orders, fills,
daily positions, cash, receivables, external flows, pending withdrawals,
account equity, performance NAV, executable weights, target/implementation/cost
drag, and a resumable checkpoint.

Whole-lot sizing maximizes deployed stock notional before minimizing target
deviation. For broad cross-sections, it preallocates all but the final four lots
around each continuous target and uses a deterministic, target-aligned heap to
fill that bounded neighborhood; ordinary cross-sections retain the exact
two-stage model. A HiGHS numerical solver error on the small-universe path
triggers one retry of the identical model with presolve disabled. The retry
keeps the objective, bounds, budget, and integrality constraints; it does not
relax infeasibility or accept a partial solution. Rounded quantities must still
satisfy the stock budget and the first-stage deployment floor.
