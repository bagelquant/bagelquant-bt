# BT AI workflow and rule routes

Read [development](rules/development.md) first, then only affected topic rules.

| Topic | Rules | Docs |
| --- | --- | --- |
| Signed plans, actual-state sizing, prices, FIFO, lots, costs, checkpoints | [Account/execution](rules/account-execution.md) | [Execution](../docs/en/reference/account-backtest.md), [costs](../docs/en/reference/transaction-costs.md) |
| Alpha/weights, IC, maturity, horizons, comparisons, inference | [Evaluation/statistics](rules/evaluation-statistics.md) | [Alpha](../docs/en/reference/factor-evaluation.md), [inference](../docs/en/reference/research-statistics.md) |
| Public API, cache/receipts, local workers and resource boundaries | [Development](rules/development.md) | [Architecture](../docs/en/architecture.md), [API](../docs/en/reference/public-api.md), [resources](../docs/en/performance.md) |

Stage 4 implements three evaluation modes and BT-owned result storage. Removed
signal/policy/account runners have no compatibility path. Core owns numerical
values/graphs/model evidence; Data owns datasets/PIT/frozen inputs; Workbench
composes public APIs and retains China semantics, governance, app relationships,
backend references and global scheduler policy. Broader Workbench cleanup is
stage 5; real database/service cutover is stage 6 with separate authorization.

Workspace discovery and standalone fallback are in [AGENTS](../AGENTS.md).
Versioned rules and source travel through Git; ignored full-task records remain
in the verified workspace's .ai/tasks. Historical task plans are evidence, not
current API authority. Package docs remain owned here and are not website input.
