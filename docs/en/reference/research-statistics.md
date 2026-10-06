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
