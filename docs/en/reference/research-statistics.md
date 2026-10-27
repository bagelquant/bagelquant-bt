# Saved research statistics

Public functions consume explicit saved tables and never fit models, fetch
providers or change governance. `return_statistics` owns return, volatility,
Sharpe, drawdown, Calmar and hit-rate summaries. Risk fit/profile/linking functions
and exposure/factor functions reuse one numerical formula per concern.

HAC inference and `library_endpoint_tests` apply two-sided p-values and BH across
every valid declared trial, including abandoned trials. Missing or unsupported
samples retain an unavailable reason. `deflated_sharpe` requires explicit trial
scope and a common finite sample; correlation/effective-trial limitations remain
visible. Workbench selects protocols, trial families and governance policy.

`common_sample_comparison`, `partial_rank_ic`, `incremental_ic_summary`,
`factor_return_correlation` and `holdings_factor_exposure` consume matching saved
coordinates/returns/holdings. `account_fill_turnover` uses actual fill notional
and prior equity. Default/custom periods aggregate saved primitives and correct
maturity without account or model replay.

`capacity_participation` uses strictly prior 20 market-session traded amounts;
execution-day observations are excluded. Nonnegative known zero is valid;
missing/negative/nonfinite coverage or a zero denominator is unavailable. This
is diagnostic capacity, not a fill limit. `execution_stress_scenarios` supplies
nine native-rule one-factor cases with independent accounts.

## Effective sample and downside risk

`evaluation_sample(frames, start=..., end=..., minimum_history_years=None)` reports requested/effective dates, first signal/economic dates, observation counts, leading exclusions, missingness and warnings. Signal coverage is authoritative when supplied; otherwise finite saved economic observations set the start after maturity filtering. Zero returns are valid. Subsequent gaps stay in the calendar. Calendar-history thresholds are caller policy.

`return_statistics` also exposes `expected_shortfall_95` (gross/net prefixes when available). It is minus the average of the worst `ceil(0.05*N)` finite returns, with at least 20 observations, not an annualized quantity. An entirely profitable sample can produce a negative ES, correctly reflecting gains in that empirical tail. Undefined metrics retain a reason. Saved research-window aggregation exposes annual/recent performance, annual IC and coverage summaries; recent means cutoff year and preceding four calendar years.


## Coverage-qualified sample starts

`evaluation_sample(..., minimum_signal_coverage=...)` optionally requires a finite
signal count divided by the full caller-supplied Universe count to reach a
threshold in `(0, 1]`. The first qualifying date is inclusive. Empty Universes do
not qualify, finite zero signals count, and later gaps or lower coverage remain
in the sample. Search the supplied complete coverage history before clipping to
the requested window; a later low-coverage window does not choose a new start.
Without a qualifying date, the sample is unavailable and returns cannot supply
a fallback start. Omitting the argument preserves the original frozen policy.

This sample function selects dates; it does not restart an account. Callers that
require a fresh diagnostic must supply a calendar and signal values beginning at
the qualifying date to `evaluate_alpha`, with no prior diagnostic state. Saved
window aggregation accepts resolved `sample_metadata` so all sections use the
same root start. Changing a display range only aggregates saved paths.

A late-start `rolling_risk_profile` can receive `factor_catalogue` with original
factor family and first-seen dates. This preserves causal population metadata
without admitting pre-start regression observations. Warm-up counts begin with
the supplied calendar; continuation still requires original state and proofs.

`regression_min_periods=None` requires a full regression window. Hosts can choose
an earlier minimum, such as 60 inside the previous 240 sessions. Only finite,
matured joint rows train the regression, and calendar gaps retain their position.
The resolved minimum participates in policy `rolling_risk_profile.v3` and saved
continuation identity. See [Daily risk updates](../risk-updates.md).
