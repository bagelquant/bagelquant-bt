# BagelQuant Bt agent entrance

Read [CONTRIBUTING.md](CONTRIBUTING.md), [README.md](README.md) and
[pyproject.toml](pyproject.toml), then the affected technical contracts routed
by the contributor guide. [Owner boundaries](docs/en/engineering/development.md) remain authoritative.

BT owns financial execution, accounts, evaluation/statistics and BTStore artifacts. It depends only on Core. Callers supply aligned returns, economic intervals, prices, market policy and resource ceilings; evaluations consume saved values.

Inspect tracked and untracked Git changes before editing. Preserve unrelated
work, Git metadata and historical evidence. Use Python 3.13 and uv; tests use
isolated temporary roots and synthetic inputs or fake providers.
Run the applicable owner checks in CONTRIBUTING.md and report actual results.
Publication, commits, provider calls and real-data/service changes require
explicit authorization. Keep generated references owned by their generators.
