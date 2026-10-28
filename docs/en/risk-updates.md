# Daily risk updates

Public `RISK_UPDATE_POLICY` identifies the daily fit, economic maturity and carry
contract. Hosts bind this version in completion and cache identities; a change
to these numerical semantics advances it independently of Core operator versions.
The current policy is `daily_risk_maturity.v2`, including actual-sample industry
bases, independent fit flags, activity inventory and compact calendar coverage.

`plan_risk_factor_update(calendar, start=..., through=...,
completed_through=..., affected_available_date=...)` uses owner-verified metadata
to select new or affected maturity dates. A completed plan performs no numerical
work. `plan.windows(sessions=32)` exposes bounded batches independent of months.

`evaluate_risk_factor_returns(exposures, prices, plan, styles=...)` calls the
authoritative daily fit with the selected weights. Decision-day exposures join the return
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
grid. Finite complete-case samples determine the industry's fitted basis and
identification constraint; the economic maturity convention is unchanged.

## Partial regression history

`rolling_risk_profile` and `plan_risk_profile_update` accept
`regression_min_periods`. `None` requires the full `regression_window`, preserving
the generic default. An explicit integer between 2 and the window size permits
an earlier fit after that many usable joint observations. The numerical policy
is `rolling_risk_profile.v4`; state and continuation identities include the
resolved minimum, so policies cannot share incompatible continuation state.

The window still counts calendar sessions. Filter its training rows to finite
joint factor/return values whose labels have matured by the decision date,
then standardize and regress only those rows. Missing rows never extend the
lookback, train the estimator, or contribute to its sample count. For example,
a host may request a 240-session window and a minimum of 60; 59 usable pairs
remain unavailable, while 60 admit a fit if the other numerical checks pass.


## Independent fits and continuous publication

The same `fit_risk_factor_returns` / `evaluate_risk_factor_returns` kernel accepts
`industry_controls=False` for an intercept-and-styles fit without an industry
column. With industry controls, an empty style list fits intercept and industries
only. The default sqrt-cap objective and cap-weighted zero-sum constraint use
the actual complete sample. Explicit weighting="equal" uses equal objective
weights and a stock-count-weighted zero-sum constraint, requiring no caps. It
gives the stock mean as Market and industry group means minus Market as industry
returns. An industry represented in the population but missing all required
observations remains unavailable. Nonfinite coefficients are rejected.

Daily updates return `industry_activity` independently of fit success, with dated
market anchors and each represented industry's eligible/sample counts. The host
persists that original inventory with independent family source, fit and state
receipts. `require_continuity=True` retains the first supported start and rejects
later failures or missing mature profile observations. Leading warm-up and
unmatured tails remain distinct. The flag participates in continuation policy.

BTStore captures date/maturity evidence when publishing a factor-activity table
with Date `time`, String `factor` and Boolean `active`, without `asset_id`.
The full typed row hash binds activity and eligible/sample counts; ordinary
profile proofs read these metadata summaries without decoding activity rows.
For an existing table without evidence, explicitly use `index_plan(evidence=True,
references=[activity])` and `build_index`, preserving the original bytes and
receipts. Published chapters remain immutable: invalidation does not permit
replacing a chapter with different contents. An authorized complete store reset
is a separate operation.

A rolling profile supplied `industry_activity` fits the union of industries
active in its training window, omitting inactive terms from each day's equation.
Moments use one admitted N-session calendar, accumulated from active observations
and intersections, without manufacturing zero-valued factor-return observations.
Centering includes the declared activity mask. Currently inactive controls have
no published beta/contribution; new controls without identifiable matured
variation have warm-up/null betas. Returns contradicting activity, incomplete
activity and missing active data cannot establish strict completion.

`link_risk_contributions(..., require_continuity=True)` rejects interior gaps;
it never silently restarts cumulative wealth after an unavailable interval.
