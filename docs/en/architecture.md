# Architecture and design

## Saved values and evaluation

BT consumes complete saved Alpha/Prediction values and Portfolio targets.
Value calculation, selection, generic optimization and training belong to Core;
BT owns diagnostics, account simulation, statistics and Plotly. Evaluation never
builds upstream values.

```text
Saved Prediction + explicit market inputs → diagnostics / metrics / figures
Saved weights + rebalance state + market inputs → account → returns / holdings / fills / figures
```

## Diagnostic chapters

run_daily_prediction_sections shares prepared signals, calendars, ranks and
prices and executes only required components. It aggregates one label window
at a time and saves numerical primitives/statistics rather than huge future-
return matrices. Incremental labels reuse verified prefixes and revisit cross-
boundary windows, newly mature labels and historical autocorrelation pairs.

## Account simulation

evaluate_portfolio_targets consumes explicit rebalance/hold/unavailable decisions.
Target zeros exit positions; hold issues no new target; unavailable retains its
reason. Accounts support integer lots, cash, costs, available quantities, T+1,
blocks, corporate actions and pending execution. Rules are caller-provided and
never guessed from stock codes. Filled holdings may differ from targets; both
are stored separately.

## State and package boundaries

Account checkpoints retain positions, cash, pending execution and corporate-
action state. The caller proves causal prefixes of predictions, targets, prices,
constraints, actions and settings before continuation. Failure leaves valid old
results intact. Legacy policy adapters preserve frozen monthly behavior and
delegate generic weight optimization to Core. BT depends only on Core; it imports
neither Data nor Workbench. Callers own data, definitions, persistence and UI.
