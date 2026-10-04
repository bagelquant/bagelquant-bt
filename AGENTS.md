# BT agent entry

The refactor target for this repository is typed scheduling, saved-target
account simulation, transaction costs, prediction evaluation, statistics,
results/reports, Plotly and its account/evaluation artifact persistence.
Depend on Core for Panel/Prediction types and numerical primitives; never import
Data or Workbench, fetch data or assume China-market rules in generic APIs.
The staged target and current implementation baseline are distinguished in
[development rules](.ai/rules/development.md); step 1 changes instructions only.

Before work, read [`.ai/README.md`](.ai/README.md), mandatory development rules,
affected topic rules, [`README.md`](README.md), [`pyproject.toml`](pyproject.toml)
and relevant local docs. Inspect tracked and untracked Git changes first.
Keep this entry short; detailed owner contracts live under `.ai/rules/`.

- Use Python 3.13 and `uv`; run commands from this repository.
- Preserve unrelated work, Git metadata, credentials, environments and data.
- Tests use synthetic inputs and isolated temporary roots; never read/mutate
  the workspace's real data root or call real providers for validation.
- Do not commit, push, create PRs, merge, release, deploy, install services,
  update real data or change governance unless explicitly requested.
- Preserve the typed `PredictionPanel` prediction-backtest boundary and the
  separate explicit saved-target account APIs. Composition returns raw typed
  predictions; express normalization explicitly downstream.
- Evaluation consumes saved values/targets and never builds/trains upstream.
  Actual broker connectivity, order planning and live submission are out of scope.

For formal work in an integrated workspace, create/resume a root `.ai/tasks/`
record using the workspace CLI. Discover the workspace with
`git rev-parse --show-superproject-working-tree`. If empty, inspect checkout
ancestors as candidates. Accept only a candidate that is its own Git root,
declares the six component paths in `.gitmodules` and has their index gitlinks
(mode `160000`); this checkout must match its exact declared relative owner path.
For every candidate, also verify
root `AGENTS.md`, `.ai/README.md`, `.ai/workflow.md`, `.ai/rules/workspace.md`
and `scripts/ai.py` exist. Read root workflow and workspace rules; load
cross-repository contracts only for affected topics. Never guess parent paths.
In a standalone checkout, follow these local rules and record plan, validation
and handoff in the conversation; do not create a component task directory.
Plan/read-only mode never writes task records or implementation files.

Validate code changes with `uv run ruff check .` and `uv run pytest`.
Report affected contracts, checks/results, unrun checks and next steps honestly.
