# Daily risk updates

Public `RISK_UPDATE_POLICY` identifies the daily fit, economic maturity and carry
contract. Hosts bind this version in completion and cache identities; a change
to these numerical semantics advances it independently of Core operator versions.

`plan_risk_factor_update(calendar, start=..., through=...,
completed_through=..., affected_available_date=...)` uses owner-verified metadata
to select new or affected maturity dates. A completed plan performs no numerical
work. `plan.windows(sessions=32)` exposes bounded batches independent of months.

`evaluate_risk_factor_returns(exposures, prices, plan, styles=...)` calls the
authoritative constrained daily WLS. Decision-day exposures join the return
between the next two session opens; the return becomes available at the second
open. Missing sessions retain calendar position. Carry one preceding actual
price observation per asset, including suspensions. Returned state records the
completed boundary, preceding-session carry boundary and two pending decisions.

The result exposes `factor_returns`, `diagnostics`, `price_carry`, `state` and
actual `fit_count`. Core owns canonical stock/style exposure artifacts; BTStore
owns fit/diagnostic/carry tables and state metadata. Publish immutable delta
tables, compose registered original references, and close with `publish_shared`
using its expected-receipt CAS. Ordinary append need not read old fit rows.

The declared population is the input decision-coordinate population, unless
an explicit `population` is supplied. A host using the causally observed union
retains null withdrawal coordinates and excludes unobserved coordinates; its
eligible/excluded diagnostic counts differ from an earlier rectangular monthly
grid. Finite complete-case samples, coefficients, constraints, rank checks and
economic maturity formulas remain unchanged.
