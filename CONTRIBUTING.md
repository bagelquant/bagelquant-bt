# Contributing to BagelQuant Bt

BT owns financial execution, accounts, evaluation/statistics and BTStore artifacts. It depends only on Core. Callers supply aligned returns, economic intervals, prices, market policy and resource ceilings; evaluations consume saved values.

## Setup and supported platform

Use macOS, Python 3.13 and uv. Read [README.md](README.md),
[pyproject.toml](pyproject.toml) and the affected technical contracts below.
Install the declared development dependencies with `uv sync --locked`.
The editable Core dependency expects a public Core checkout at `../bagelquant-core`, as declared in pyproject.toml. No private integration checkout is required.
Run commands from this repository. Validation runs locally; GitHub Actions is disabled.
This contributor guide and the package's tests are available in a standalone checkout.

## Technical contracts by topic

The engineering documents specify owner invariants; the linked manuals explain
usage. Keep one authority for each formula, schema, model and execution path.
Verify a disagreement against current source and representative tests before
updating the affected documentation. Historical evidence keeps its original meaning.

| Topic | Rules | Docs |
| --- | --- | --- |
| Signed plans, actual-state sizing, prices, FIFO, lots, costs, checkpoints | [Account/execution](docs/en/engineering/account-execution.md) | [Execution](docs/en/reference/account-backtest.md), [costs](docs/en/reference/transaction-costs.md) |
| Alpha/weights, IC, maturity, horizons, comparisons, inference | [Evaluation/statistics](docs/en/engineering/evaluation-statistics.md) | [Alpha](docs/en/reference/factor-evaluation.md), [inference](docs/en/reference/research-statistics.md) |
| Public API, cache/receipts, local workers and resource boundaries | [Development](docs/en/engineering/development.md) | [Architecture](docs/en/architecture.md), [API](docs/en/reference/public-api.md), [resources](docs/en/performance.md) |

## Engineering and validation

Inspect tracked and untracked Git changes and preserve unrelated work and Git
metadata. Keep changes focused, dependencies explicit, typed boundaries validated,
ordering deterministic, and errors actionable. Use logging in library/application
logic and vectorized Polars/NumPy where appropriate. Remove obsolete paths and
their exports/tests/docs together; add compatibility only for a concrete requirement.

Use isolated temporary roots and synthetic inputs or fake providers for tests.
Keep test fixtures, caches and output outside the checkout. Preserve real data,
credentials, authored research, historical receipts and recovery evidence.
Performance work preserves numerical/PIT contracts. Formula changes need a
hand-checkable example and a simple reference; cover sparse/missing/duplicate/empty
inputs. Calculation changes must prove bounded new/affected work and historical
reuse under the owner contracts. Distinguish observation, information cutoff,
signal-effective and execution dates; keep stable typed keys and missingness.

For code changes, run the applicable checks:

```bash
uv run ruff check .
uv run pytest
```

Choose focused checks for prose changes. Test the owner before affected consumers
when a public boundary changes, and report actual results and unrun checks.
Regenerate generated references, catalogs, OpenAPI and API types from their owner;
never edit them manually. Update the owner documentation with public behavior,
formula, schema and workflow changes. Active authored documentation is English;
preserve original sources, market identifiers, Unicode data and historical evidence.

Keep main stable and branches focused. Use imperative Conventional Commit
summaries of at most 72 characters. Keep credentials, datasets, databases,
environments and build output out of Git. Commits, pushes, PRs, merges, releases,
deployment, services, provider calls and real-data/governance changes require
their own explicit authorization; validation does not authorize those operations.
