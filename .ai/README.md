# BT AI workflow and rule routes

Always read [development rules](rules/development.md). Then load affected topic
rules; rules are authoritative instructions and linked docs explain use.

| Task topic | Owner rules | Relevant docs |
| --- | --- | --- |
| Signal/Prediction, scheduling, target policies, prices/costs, lots, checkpoint continuation | [Account and execution](rules/account-execution.md) | [Architecture](../docs/en/architecture.md), [Account engine](../docs/en/reference/account-backtest.md), [Transaction costs](../docs/en/reference/transaction-costs.md) |
| IC/quantiles, horizon/path diagnostics, HAC/BH/DSR, comparisons, capacity/stress | [Evaluation and statistics](rules/evaluation-statistics.md) | [Factor evaluation](../docs/en/reference/factor-evaluation.md), [Research inference](../docs/en/reference/research-statistics.md), [Performance](../docs/en/performance.md) |
| Public exports/import cost, shared primitives, application contract | Both owner rules and [development](rules/development.md) | [Public API](../docs/en/reference/public-api.md), [Internals](../docs/en/reference/internals.md) |

BT is independently versioned and depends only on Core. Core owns generic
numerical processing/training/optimization; Workbench owns data resolution,
market rules, definitions, governance, persistence, lifecycle and orchestration.
BT documentation is collected by the website from GitHub default branches,
independently of workspace gitlinks; edit docs here, not generated website content.

Integration discovery and standalone fallback are in [AGENTS.md](../AGENTS.md).
After verifying an integration root, use its `.ai/workflow.md` and root task CLI;
the bilingual usage guide is `docs/ai-workflow.md` / `docs/zh-CN/ai-workflow.md`
in that verified root. Rules/templates are versioned; actual task records are
local, ignored, and owned only by the workspace root. A clone does not restore them.
