# Public API — BT 0.12

- `evaluate_alpha(alpha, forward_returns, ...)` accepts a saved Core numeric or
  prediction Node. `evaluate_weights(weights, forward_returns, ...)` accepts a
  saved weights Node. Both return tables/metrics and the weight mode a checkpoint.
- Returns require time, asset_id, forward_return, interval_start, interval_end
  and available_date. Time is the caller's alignment key; economic intervals
  must be sequential/nonoverlapping. No implicit execution lag/price convention.
- `build_alpha_weights(alpha, method="book"|"spread", every=..., calendar=...,
  anchor=...)` builds gross-one/net-zero weights from latest causal snapshots.
- `evaluate_execution(transactions, market_prices, initial_capital=..., ...)`
  consumes stable plan_id, execution_date, asset_id and signed integer quantity.
  Prices explicitly provide execution_price/valuation_price or open/close.
  Optional per-plan reference_price overrides the market reference_price;
  otherwise the ideal account uses the supplied execution price.
- `run_execution_from_weights(weights, decisions, market_prices, calendar=...,
  session_lag=..., initial_capital=..., ...)` uses actual decision-close state
  to freeze deltas; rebalance/hold/unavailable stay explicit. Frozen shares are
  never resized at the execution price. Decisions have time/status columns.
  reference_price_mode="execution" is the default; "decision" explicitly freezes
  decision-close reference prices. Workbench chooses "decision".
- `save_checkpoint`, `load_checkpoint`, `merge_execution_tables` and
  `summarize_transaction_pnl(frames, through=...)` own continuation and saved
  historical account views. New record-date discoveries require verified actual
  and reference position/session/lot histories. No historical replay on reads.
- `ExecutionConfig` owns neutral money costs/lots/settlement/retry/annualization.
  `ExecutionCostRule` requires stable ID/version/JSON parameters and a pure valid
  monotone-cost quote returning `ExecutionCostQuote`.
- `return_statistics` and the public financial/risk/inference functions operate
  on explicit saved tables. BT defaults annualization 252 and quantiles 10;
  Workbench explicitly chooses 240 and China market rules.
- `BTStore`, `evaluation_identity`, `BTExecutionOptions` and
  `run_evaluation_batch` own result persistence and bounded local execution.

No public evaluation fetches providers, trains/builds upstream graphs, probes
hardware or changes governance. [Execution](account-backtest.md),
[costs](transaction-costs.md) and [architecture](../architecture.md) define details.
