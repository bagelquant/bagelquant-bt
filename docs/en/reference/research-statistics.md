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

`quantile_rank_information_coefficients(..., periods=...)` compounds each
quantile's saved daily returns over the same `[time, next_time)` interval.
Every quantile needs one finite return on every date observed in that interval;
an absent row has the same unavailable effect as an explicit null or nonfinite
return. For example, if Q1 and Q2 have two daily returns but Q3 has only one,
the period's Rank IC is unavailable. It never compares Q3's shorter holding
period with the other quantiles' complete periods. Dates absent from all supplied
quantiles cannot be inferred; callers supply the complete daily observation axis.
Aggregation reads saved primitives only and leaves historical results unchanged.

`compare_portfolio_to_benchmarks` compounds each matched Portfolio/benchmark
sample from wealth 1. Relative wealth divides those two paths; relative drawdown
divides that ratio by its running peak, including the initial ratio 1, then
subtracts 1. A Portfolio return of -10% against a flat benchmark therefore
starts at -10% relative drawdown. An initial complete Portfolio loss remains
-100% relative drawdown instead of an undefined zero-over-zero ratio.
`BENCHMARK_COMPARISON_POLICY` identifies this derived policy as
`saved-value-benchmark.v2`. Hosts that persist comparisons bind the policy to
their Evaluation/chapter identity and retain old receipts. Re-aggregation uses
saved account returns; it does not authorize account replay or value rebuilding.
Each selected window starts with its own unit base and excludes earlier peaks.

`build_universe_benchmark_returns` requires unique, nonnull `(time, asset_id)`
coordinates after its supported Date/String normalization. Duplicate stock
returns are rejected rather than counted twice or silently deduplicated. Missing
or nonfinite return values still reduce observed coverage and use the existing
available-sample renormalization; they are distinct from missing coordinate keys.

`capacity_participation` uses strictly prior 20 market-session traded amounts;
execution-day observations are excluded. Nonnegative known zero is valid;
missing/negative/nonfinite coverage or a zero denominator is unavailable. This
is diagnostic capacity, not a fill limit. `execution_stress_scenarios` supplies
nine native-rule one-factor cases with independent accounts.

## Annualized metrics and Alpha Return comparisons

Sharpe uses the arithmetic daily mean divided by sample standard deviation
(`ddof=1`), multiplied by `sqrt(A)`, with zero risk-free return. `A` is the
caller's frozen annualization. Annualized volatility uses the same square-root
factor; annualized return compounds wealth over the finite observation count.
ICIR exposes both raw and annualized fields. For a horizon covering `w` market
sessions, `annualized_icir = icir * sqrt(A / w)`; bucket width includes both ends.

`alpha_return_comparison(lag_returns, annualization=A)` returns `(metrics,
tables)` from saved Date `time`, `path_kind` (Book/Tail), integer `lag`, and
Gross/Net return rows. Apply authorization and maturity bounds first. It retains
shared lag dates and derives Book/Tail table statistics and daily drawdown inputs
from lag zero, using `return_statistics`. An absent/empty lag-zero sample stays
unavailable; full-path returns cannot supply a fallback. Null/nonfinite returns
remain unavailable observations and genuine zeros remain valid. Saved causal
returns are never restarted or simulated at a window boundary.

`metrics.comparison_sample` identifies the comparison's `start`, `end`,
`observations`, `annualization` and `basis="common_lag_dates"`. The ordinary
`sample` still describes root signal coverage. `aggregate_saved_window` uses
this comparison for the Alpha Return section; other sections keep their own
declared samples.

`annualized_turnover(mean_daily_turnover, annualization=A)` multiplies the mean
daily full-L1 turnover by `A`. The result is a yearly fraction: `3.0` means
300%/year. The caller selects observations and excludes initial entry when its
protocol requires that exclusion. This does not divide full-L1 turnover by two.
Daily curves, IC, hit rates, drawdowns and daily expected shortfall keep their
original units; they are not annualized ratios.

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
The resolved minimum participates in policy `rolling_risk_profile.v4` and saved
continuation identity. See [Daily risk updates](../risk-updates.md).


Hosts can supply causal `industry_activity` and enable `require_continuity` for
masked rolling regressions. Inactive terms are omitted without inventing return
rows. Missing active observations or mature Gross block after the supported
start. Industry-only and style-only fits share the public WLS kernel.
