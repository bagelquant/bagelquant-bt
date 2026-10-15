# BT agent entry

BT owns portfolio mechanics, accounts, costs, evaluation/statistics and their
artifact identity, storage, receipts, invalidation and recovery. Public modes are
`evaluate_alpha`, `evaluate_weights` and `evaluate_execution`; the saved-weight
execution bridge uses the same authoritative signed-share engine.
Calculation changes must incrementally update new/affected daily primitives and
maturity tails, reuse saved chapters/account state and check prerequisites from
records; prove historical reuse under the evaluation/account owner rules.
`alpha_update_plan` selects original preceding signal checkpoints from compact
saved metadata and affected intervals. Missing safe state blocks repair;
the same `evaluate_alpha(previous=...)` engine performs the admitted suffix.
Explicit legacy input maintenance preserves same-boundary signal primitives
under complete source-owner attestations; it grants no native continuation state.
Native account continuation separately validates complete source prefixes before
tail reads, retains cumulative state and publishes original daily references.
`plan_risk_factor_update` plans only newly mature/affected daily fits from
metadata; `evaluate_risk_factor_returns` owns WLS, maturity and price carry.
Core owns style-exposure graphs; BTStore owns fit/diagnostic/state receipts.
Risk profile policy v2 admits only causally visible factors and uses metadata-only
`plan_risk_profile_update` plus the same `rolling_risk_profile(previous=...)`
formula for bounded daily updates. Saved specific-return halos and immutable
delta references belong to BT; unknown prefix/state never triggers history replay.

Before work read [`.ai/README.md`](.ai/README.md), mandatory development rules,
affected topic rules, [`README.md`](README.md), [`pyproject.toml`](pyproject.toml)
and relevant local docs. Inspect tracked and untracked Git changes first.

- Depend only on Core. Never import Data/Workbench, fetch providers or choose
  China rules. Saved evaluation never builds or trains an upstream graph.
- Require caller-aligned forward returns with explicit economic interval and
  availability dates; never add an implicit lag/price convention.
- Keep portfolio mechanics separate from reusable evaluation functions. Maintain
  one account loop, one numerical formula and one artifact authority per concern.
- Workbench supplies global admission and resources; BT exposes explicit workers
  and enforces caller budgets without probing the machine.
- Ordinary chapter/table reads trust registered original receipts, preserving row
  counts and indexed schemas without byte/logical rehashing. `verify` is the full
  byte/logical/index audit. Registered reference-only publication avoids decoding;
  new outputs retain canonical hashes. Explicit initialization enables WAL.
- Use Python 3.13 and uv. Tests use synthetic inputs and temporary stores; never
  touch real shared data, providers, authored work or historical receipts.
- No commit, push, PR, merge, release, deployment, service, real data update or
  governance transition without explicit authorization. Breaking cleanup does
  not authorize deleting historical evidence or database/service cutover.
- Read-only/Plan modes never write task records. Formal authorized integrated
  work uses the workspace full-task workflow and independent review.

Find an integrated workspace through Git superproject metadata or verified
ancestors: its own Git root must contain root AGENTS, .ai/README/workflow/rules,
scripts/ai.py, .gitmodules and mode-160000 gitlinks for the four packages, and
this checkout must match its declared owner path. Without that verified root,
follow local rules and record progress in the conversation; never create another
component task store.

Validate with `uv run ruff check .` and `uv run pytest`; report actual results,
unrun checks, affected consumers and remaining migration gaps honestly.

Alpha horizon windows also accept this explicit budget through `evaluate_alpha`
options; execution evidence stays separate from numerical metrics and identity.

`evaluation_sample` supplies neutral effective dates/counts/missingness from saved signal/economic observations; callers supply minimum calendar-history policy. Leading unavailable history is excluded, genuine zero returns and subsequent gaps retained. Return statistics expose empirical daily ES95 with explicit sample requirements.

Caller native admission wraps complete mixed evaluation work and keeps its
inner execution serial; standalone BT retains explicit bounded threads. Ready
slots refill after any completion while result order and cancellation persist.
