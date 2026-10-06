# bagelquant-bt

BT 0.12 provides standalone financial evaluation of saved values, portfolio weights and signed
share transaction plans. BT depends only on Core; callers supply all returns,
prices, calendars, market rules and receipt references.

| Mode | Public entry | Inputs and result |
| --- | --- | --- |
| Alpha / Prediction | `evaluate_alpha` | Saved Core numeric/prediction Node; caller-aligned one-session forward returns; IC, rank IC, persistence, horizons, quantiles and Book/Spread diagnostics |
| Portfolio weights | `evaluate_weights` | Saved Core weights Node; forward returns; independent Gross/Net normalized NAV, drift, turnover and lag diagnostics |
| Portfolio execution | `evaluate_execution` | Stable `plan_id`, execution date, asset and signed integer share quantity; explicit prices and initial capital; cash/holdings, fills, FIFO costs and ideal-versus-actual P&L |

`build_alpha_weights` produces gross-one, net-zero centered-rank Book or tail
Spread weights. Caller-selected rebalance cadence uses the latest causal whole
snapshot. `run_execution_from_weights` converts saved targets into frozen deltas
at actual decision-close state using the same execution engine.

Research costs default to `0.0005 * full-L1 weight change` and subtract directly
from period returns. Holdings drift between rebalances; Gross and Net maintain
independent state. This is a virtual research NAV, with no money minimum fee.
Execution costs default to `max(0.0005 * executed notional, 5)` per order and
execution date, with lot 1, settlement 0 and no tax/slippage. Unfilled remainders
expire unless retry is explicitly requested. The zero-cost ideal plan must be
cash- and inventory-feasible; otherwise evaluation fails.

BT does not choose a lag or a return price convention. Forward-return rows
include `time`, `asset_id`, `forward_return`, `interval_start`, `interval_end`
and `available_date`. The caller owns the label alignment and maturity cutoff.
Workbench's China adapter labels decision t with adjusted t+1 open to t+2 open
and explicitly supplies its market rules and annualization 240; BT defaults to
252 and ten quantiles.

`BTStore(meta_path, artifact_path)` owns SQLite metadata, immutable Parquet
artifacts, receipts, integrity, invalidation and recovery. Consumers use public
read/publication APIs and retain opaque receipt references. No Workbench database
or backend-file access is required for standalone evaluation or caching.

`BTExecutionOptions` and `run_evaluation_batch` accept explicit worker and Core
resource limits for independent evaluations. One account is causal and sequential;
BT never probes RAM/CPU or allocates a global budget. Workbench owns admission
and machine policy.

See [quick start](docs/en/quick-start.md), [public API](docs/en/reference/public-api.md),
[execution](docs/en/reference/account-backtest.md), [costs](docs/en/reference/transaction-costs.md),
and [中文文档](docs/cn/index.md). AI contributors start with [AGENTS.md](AGENTS.md).

```bash
uv run ruff check .
uv run pytest
```

The stage-4 contract is breaking: removed signal/policy/account runners have no
aliases or migration readers. Real database/service cutover remains separately
authorized stage 6; historical/authored evidence is preserved.
