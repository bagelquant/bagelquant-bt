# BT AI workflow and rule routes

Always read [development rules](rules/development.md), including the staged
refactor target and current baseline. Then load affected topic rules; linked
docs describe current use until the corresponding code stage updates them.

| Task topic | Owner rules | Relevant docs |
| --- | --- | --- |
| Signal/Prediction, scheduling, target policies, prices/costs, lots, checkpoint continuation | [Account and execution](rules/account-execution.md) | [Architecture](../docs/en/architecture.md), [Account engine](../docs/en/reference/account-backtest.md), [Transaction costs](../docs/en/reference/transaction-costs.md) |
| IC/quantiles, horizon/path diagnostics, HAC/BH/DSR, comparisons, capacity/stress | [Evaluation and statistics](rules/evaluation-statistics.md) | [Factor evaluation](../docs/en/reference/factor-evaluation.md), [Research inference](../docs/en/reference/research-statistics.md), [Performance](../docs/en/performance.md) |
| Public exports/import cost, shared primitives, application contract | Both owner rules and [development](rules/development.md) | [Public API](../docs/en/reference/public-api.md), [Internals](../docs/en/reference/internals.md) |

BT is independently versioned and depends only on Core. The target assigns
generic numerical processing/training/optimization and graph artifacts to Core,
account/evaluation artifacts to BT, and dataset/frozen-input APIs to Data.
Workbench keeps China semantics, authored definitions, application metadata,
governance, lifecycle and orchestration, consuming backend public APIs.
Current result persistence remains partly in Workbench pending stages 4 and 5;
the topic rules describe that baseline, not a completed ownership transfer.
BT documentation is collected by the website from GitHub default branches,
independently of workspace gitlinks; edit docs here, not generated website content.

Integration discovery and standalone fallback are in [AGENTS.md](../AGENTS.md).
After verifying an integration root, use its `.ai/workflow.md` and root task CLI.
For package-boundary work also read its
`.ai/rules/contracts/package-boundaries.md`. Local target rules remain sufficient
in a standalone checkout.
The bilingual usage guide is `docs/ai-workflow.md` / `docs/zh-CN/ai-workflow.md`
in that verified root. Rules/templates are versioned; actual task records are
local, ignored, and owned only by the workspace root. A clone does not restore them.
