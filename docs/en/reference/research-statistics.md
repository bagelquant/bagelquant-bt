# Research inference and comparison

These public primitives consume explicit saved tables. They do not fit models,
simulate accounts, select production candidates or manage validation batches.
Workbench freezes protocols, trial families, source receipts and result references.

`library_endpoint_tests(trials, q_max=0.10)` uses two-sided HAC p-values from each
declared primary endpoint, applies BH to every valid result including abandoned
trials, and checks the declared direction and minimum effect. Missing statistics
retain null p/q values with an unavailable reason. For p-values 0.01 and 0.04,
BH q-values are 0.02 and 0.04; a failed third trial without a p-value is listed
separately and does not increase the valid-test denominator.

`deflated_sharpe(returns, target=..., trials=..., minimum_sessions=240)` accepts
long-form `(time, trial, return)` net returns. It uses one common finite sample,
retains the declared trial scope and correlation matrix, and estimates effective
trials by `rho + (1-rho)*M`. At least three trials, 240 sessions by default and
more sessions than trials are required. Constant returns, negative unsupported
average correlation, a singular/ill-conditioned correlation matrix or fewer
than two effective trials produce an explicit unavailable result. This auxiliary
estimate does not represent undocumented historical experiments. Its DSR formula
and correlation approximation follow [Bailey and López de Prado (2014)](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

`common_sample_comparison` compares scalar daily primitives on matching finite
dates. `common_prediction_ic` uses matching finite asset/date coordinates and
reports incremental and conditional rank IC; a collinear variant has no independent
signal. `account_fill_turnover` uses actual fill notional and prior equity, including
the frozen initial capital on the first session.

`capacity_participation` divides each fill's absolute CNY notional by the asset's
average CNY traded amount over the preceding 20 market sessions. Every session
needs a nonnegative finite observation, including a known zero. The execution-day
amount is excluded; missing coverage or a zero mean produces null participation.
Account stress scenarios use the existing
saved-target account engine, with independent cash, positions, fills and checkpoints.
The capital factor scales initial capital and the fixed-notional capital target;
otherwise the fixed-notional account would withdraw the increase immediately.
