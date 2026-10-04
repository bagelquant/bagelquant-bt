# Saved-value evaluation, statistical evidence and stress

## Diagnostics and saved inputs

- Evaluate complete saved roots/targets with explicit market inputs and selected
  components only. Never build values, train upstream, call providers or make
  passive reads submit computations. Callers own persistence and authorization.
- Share diagnostic weights, prepared market inputs and numerical primitives.
  Stream bounded label windows instead of retaining wide full-history matrices.
  Continue only verified causal prefixes; current freshness and historical
  integrity remain separate. Failures preserve valid prior saved results.
- Daily prediction diagnostics preserve cumulative 1/5/10/20/40/60/120-session
  and bucket windows, centered-rank Book/Tail, deterministic quantiles, lag
  diagnostics, PIT label maturity and frozen Newey-West/BH inference. These
  diagnostics never become account NAV or actual holdings.
- A missing member return contributes zero at its original Book/Tail weight,
  with visible expected/observed coverage; never reselect or renormalize using
  future label availability. Quantile/common-sample and IC rules remain strict.
- Quantile-rank IC is derived from stored gross q1-to-qN returns by the public
  BT statistic using Core's formula. Infer N from labels so historical q5 stays
  readable. Incomplete groups, no finite group return and constant group returns
  produce null.
- Saved-turnover/risk comparisons require exact interval/settings/source matches.
  Stress-only updates must not recalculate unused baseline turnover. Default and
  custom periods consume saved primitives/realized account paths; aggregation
  does not replay accounts or rebuild values.

## Inference and capacity

- Own common-coordinate/conditional IC, common-finite-date saved-return
  comparisons, two-sided HAC/BH library inference and Deflated Sharpe
  diagnostics. Include every valid trial, including abandoned trials, in
  multiplicity adjustment; unsupported evidence is unavailable with a reason.
- Capacity uses explicit CNY `turnover_amount` from strictly prior 20 observed
  market sessions, excluding execution day; it is diagnostic, never a fill limit.
  Capacity kernel v2 accepts known zero observations, rejects negative/nonfinite/
  missing observations, and leaves a zero denominator unavailable. Its chapter
  identity changes without resimulating shared accounts.

## Implementation stress

- Preserve nine one-factor scenarios: capital x2/x5/x10, commission x2/x4,
  slippage x2/x4, and session delay +1/+4. Scale initial capital and fixed-notional
  capital target together so fixed-notional flows do not withdraw the increase.
- Share prepared inputs while retaining independent accounts/checkpoints.
  Consumers visit targets and stress accounts serially with the full native-thread
  budget; release full account tables between scenarios. Stress chapters retain
  only saved return paths/capacity diagnostics; full cash/position/fill tables
  remain in the independent shared account.
- Use one total thread/memory budget, shrink later batches/admission under
  pressure and retain causal account dates. Resource limits are identity-neutral;
  numerical settings remain explicit. Read horizons only from caller-supplied
  runtime Available Date/research settings, never wall-clock guesses.
