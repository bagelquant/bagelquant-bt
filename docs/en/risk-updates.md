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

## Partial regression history

`rolling_risk_profile` and `plan_risk_profile_update` accept
`regression_min_periods`. `None` requires the full `regression_window`, preserving
the generic default. An explicit integer between 2 and the window size permits
an earlier fit after that many usable joint observations. The numerical policy
is `rolling_risk_profile.v3`; state and continuation identities include the
resolved minimum, so policies cannot share incompatible continuation state.

The window still counts calendar sessions. Filter its training rows to finite
joint factor/return values whose labels have matured by the decision date,
then standardize and regress only those rows. Missing rows never extend the
lookback, train the estimator, or contribute to its sample count. For example,
a host may request a 240-session window and a minimum of 60; 59 usable pairs
remain unavailable, while 60 admit a fit if the other numerical checks pass.
