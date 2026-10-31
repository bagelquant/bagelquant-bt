# Saved evaluation and statistical evidence

The public `drawdown` utility includes initial wealth one in the running peak,
as do saved-window summaries: an initial 10% loss is a -10% drawdown and immediate
bankruptcy is -100%. It consumes the supplied row order and retains the existing
null-as-zero path convention. `cross_sectional_factor_returns` validates unique
normalized panel keys and uses only matched finite factor/return pairs. Its
sample count is Int64 in empty and populated results; fewer than three finite
pairs or no factor variation leaves the slope unavailable. These utility fixes
do not rewrite previously saved numerical tables or chapters.

Explicit `BTStore.maintain_shared_inputs` admits retained legacy signal primitives
only at their original observation and maturity boundaries. It requires an
immutable original execution request, numerical identity reproduction, a source
report reference/SHA256 and complete typed value, Domain, price, calendar and
label projections through `SharedInputEvidence`. Source owners attest exact
consumed content (including empty coordinates, null/nonfinite values and traces);
BT checks scope/equality and retained authority, not external source semantics.
Ordinary receipt reads and the closing writer validate the registered original
chain before cache reuse. Preserve all old files, groups and numerical metadata;
an independently missing `daily_signal_coverage` may be initialized once. This
admission grants neither append nor group upgrades and cannot manufacture native
scope or continuation state. Unknown projections require explicit maintenance.

Evaluate saved Core numeric/prediction and weights Nodes or frozen execution
plans. Never build/train upstream, fetch providers or submit work during passive
reads. One pure formula owns each statistic; portfolio mechanics supply paths
and reusable evaluation functions consume explicit saved primitives.

Calculation changes require metadata-only prerequisites and actual incremental
new/affected daily work. Reuse historical primitives, chapters and causal state;
revisit newly mature horizon tails and propagate input revisions locally before
deriving current summaries from saved primitives/state. New enclosing receipts,
cutoffs and unrelated counters never justify replaying all daily statistics or
accounts. Full computation is initialization or a relevant numerical definition
change. Missing impact/prefix evidence must be an actionable plan/maintenance
requirement. Prove append/repair equality to a full reference and instrument
historical daily/account calls; an incremental action name is not acceptance.

Daily risk fitting uses `plan_risk_factor_update` and the same
`evaluate_risk_factor_returns` formula for initialization, append and repair.
Plan before loading exposures/prices; `windows(sessions=32)` bounds independent
fit batches. Decision t uses next-session-open to following-session-open,
maturing at the following session. Preserve null/complete-case exclusion,
the caller-selected weighting and constraint, rank checks
and sorted industry/asset ordering. A preceding price carry retains actual
observation dates, including suspended assets; never invent observation dates.
Core owns exposure numerical artifacts; BTStore owns returns, diagnostics and
carry/state via shared receipts. Append registered delta references and preserve
old numerical bytes, rather than compare/decode complete historical frames.
Default sqrt-cap WLS uses the cap-weighted industry constraint. Explicit equal
weighting requires no cap input and uses stock counts in that constraint; its
industry-only factors are exact group means relative to the whole stock mean.
The host declares causal diagnostic population policy; a current-day observed
coordinate union need not equal a retired rectangular monthly population.

Risk-profile policy `rolling_risk_profile.v4` uses the factor population visible
on each economic date. Future industries cannot add historical zero-exposure
rows or constant-factor counts. Explicit industry_activity proves dated activity independently of fits. Masked
regressions omit inactive terms and never manufacture factor-return observations.
Center/scaling moments use the same admitted calendar. Missing active inputs and
missing mature Gross block strict publication; new unidentifiable betas remain null. Optional regression_min_periods defaults to the full regression
window; explicit partial minima count only valid, matured joint rows inside
that calendar window. Training and standardization use that same filtered set.
The resolved minimum is bound into policy and continuation state.
`plan_risk_profile_update` checks registered prefix metadata before
reading bounded gross/factor inputs and saved specific-return halos. Append only
revisits pending maturity and new dates. A fixed-population input repair widens
at most regression_window + volatility_window - 1 sessions, selects a proved
preceding checkpoint and preserves the later unaffected references.
The saved residual halo may come from the latest original result after proving
its complete source prefix before the affected date against current inputs;
an earlier checkpoint alone does not attest residuals beyond its boundary.
Unknown impact/state, changed population outside that scope or simultaneous repair and
append require explicit planning; they never authorize full historical replay.

BT stores compact date/availability typed-row summaries and first-seen factor
catalogues in optional digest-bound table indexes at publication. Public
`reference_evidence`/`reference_interval_identity` read those records only;
missing summaries require explicit `index_plan(evidence=True)` maintenance.
Maintenance audits original bytes and never republishes numerical results.

`alpha_update_plan` accepts an EvaluationResult or its saved state mapping and
source-owned affected date intervals. It plans before reading primitive tables,
widening the affected date by requested horizon/persistence/lag dependencies and
selecting the latest preceding receipt with matching numerical policy/components.
An optional source proof callback validates complete old-cutoff prefixes using
metadata; `evaluate_alpha(previous=selected)` revalidates them when consuming.
No safe preceding state yields a blocked recovery plan, never full-history replay.
Save `pending_horizon_start` alongside path state so arbitrarily delayed labels
retain their original read scope even in metadata-only planning. Callers publish
suffix compositions from the selected original receipt, retaining old files and
the immutable receipt chain; never append repaired values to the latest stale one.

Caller-aligned one-session returns require time, asset_id, forward_return,
interval_start, interval_end and available_date. Time is an alignment label;
BT inserts no lag or price convention. Economic intervals are explicit,
nonoverlapping and sequential; availability is at/after the economic end.
Unmatured labels are unavailable, not zero. Mature missing constituents retain
fixed Book/Spread weight with visible coverage; IC and complete quantile rules
remain strict. Calendar gaps must not compress economic windows.
Periodized quantile Rank IC requires finite saved returns for every quantile on
the union of supplied dates in each execution interval. Missing quantile/date
rows remain unavailable instead of shortening one quantile's compounding window.
Group each interval once; saved-window aggregation never replays daily kernels.

Benchmark comparison policy `saved-value-benchmark.v2` includes initial relative
wealth 1 in its running peak. First-session underperformance remains drawdown;
complete initial Portfolio bankruptcy yields -100% instead of NaN. The public
`BENCHMARK_COMPARISON_POLICY` is the single policy marker for persisted default
and saved-window comparisons. Hosts bind it into new derived identities while
preserving old immutable chapters and reusing saved account/benchmark returns.
Universe benchmark inputs also require unique nonnull normalized date/asset
keys before accepting an equal- or size-weighted result. Reject ambiguous duplicates
instead of changing constituent weights or permitting coverage above one.

Alpha/Prediction share evaluation: Pearson/rank IC and IR, cross-sectional and
window returns, gross-one centered-rank Book/Spread, deterministic quantiles,
rank persistence, horizons/buckets, lag/lead-lag and rolling IC/inference.
Persistence summary policy `bt.persistence.canonical_mean.v1` uses the single
BT `summarize_signal_persistence` function: public stdlib exact-ratio finite mean,
explicit canonical NaN/infinity handling, non-null observation counts and the
existing grid/band rule. Financial daily correlations are Float64; preserve the
public summary's existing Float32 output, Float64 integer/Boolean/Decimal means,
String/Null typed-unavailable values/counts and visible non-null temporal mean's
TypeError category. All-null/empty-grid/unrequested temporal means stay unavailable.
Infer output types from public lazy schema metadata, never an old floating
reduction. Use exact Fraction adaptation for Decimal input, then the same public
stdlib mean and final output cast. Bind the
derived policy in new immutable Evaluation/chapter settings; do not bump daily
correlation or raw Alpha continuation policy. Same-boundary pre-policy state can
refresh only this summary after the ordinary complete prefix/state checks;
current-policy repeat skips reduction. Preserve original daily frame objects,
references/bytes, independent metrics and checkpoints. Never overwrite an old
chapter or claim an old summary has the new policy merely because inputs match.
Default annualization is 252 and quantiles 10. Workbench explicitly supplies
240 and its China label adapter (decision t -> adjusted t+1 to t+2 open).
Saved Alpha Return comparisons use `alpha_return_comparison`: statistics and
drawdown inputs come from the same common-lag zero rows as the lag-zero curve.
Never overwrite these with full-path statistics or fall back to full returns
when comparison history is empty. Root coverage and comparison metadata remain
separate. Sharpe consumers use `return_metrics`/`return_statistics` with frozen
annualization; ICIR charts consume the explicitly annualized horizon field.
Book/Spread paths use normalized virtual NAV with proportional full-L1 costs.
Each independent weight path preserves its causal daily state, but only its
final mature checkpoint is materialized; null-return values/observation masks
are prepared once per path. Quantile curves share columnar grouping/order
preparation while retaining the same scalar statistical formulas. Full Alpha
performance evidence covers all requested paths/windows and incremental reuse.
Immutable return preparation may use the caller's cache allowance, bounded by
one quarter of its memory target; debit both before admitting other diagnostics.
Retain exact asset axes, date/maturity windows, independent causal path state,
pre-allocation charges, zero/oversize fallback and operation cleanup. Larger
configured caches may avoid repeated panel conversion; they do not relax the
mixed-native serial inner-executor contract or increase the total budget.
Empty horizon Book/Spread/quantile counts and factor-return sample sizes use
UInt32, matching populated reductions; IC sample size remains Int64 even before
the first mature return. Appending or repairing cannot change these schemas.
At a retained-window merge, normalize only the original Int64 representation of
the four UInt32 horizon count families, with strict nonnegative/range validation.
Reject other noncanonical count dtypes. An empty retained primitive uses the
current empty schema, including IC sample size. Never rewrite original saved
tables or replay processed path sessions to normalize derived output schemas.

Pure saved comparisons, conditional/incremental IC, HAC/BH and Deflated Sharpe
remain reusable BT APIs. BH includes every valid declared trial, including
abandoned ones. Unsupported inference is unavailable with an explicit reason.
Capacity uses strictly prior 20 observed market sessions, excluding execution
session; zero amounts are valid, missing/negative/nonfinite amounts unavailable.
It is a diagnostic, never a fill limit.

Native execution stress retains capital x2/5/10, commission x2/4, slippage x2/4,
and delay +1/+4. Capital scales fixed sizing notional as well; minimum fees/tax
stay fixed for rate stress. Zero slippage remains zero. Independent accounts
share supplied read-only inputs, not cash/state. Opaque custom rules require
caller-declared scenarios rather than inferred fee components.

BTStore owns evaluation paths/checkpoints/primitives/chapters and integrity.
Factor-activity tables with Date time, String factor, Boolean active and no
asset_id receive the same typed date/maturity evidence at publication. Hash the
complete row, including eligible/sample counts. Missing old summaries require
explicit scoped owner index maintenance, never implicit reads or refits.
Workbench retains application authorization/governance and backend references.
Publication is immutable and failures preserve valid receipts; prefix append
must match full computation and revisit newly mature labels. Source changes
invalidate the corresponding current result without rewriting historical bytes.
Default/custom periods use public saved-window aggregation and cutoff-safe FIFO
summaries; no account/value/model replay or missing-evidence creation on reads.

Alpha horizon windows may run independently under caller execution options;
merge frames in requested window order before global summaries, BH/HAC and
rolling inference. Cancellation checks separate window stages and global
aggregation. No worker mutates shared inputs or publishes artifacts.
Same-boundary `up_to_date` results retain their original primitive objects and
state. Consumers reuse original references while publishing any newly proved
input binding under ordinary fences; they must not turn the read halo into
new output. `derived_refresh` replaces only the changed derived summary.

Signal persistence keeps ranks after the common-asset join and collects each
bounded pair chunk as one query. Its public wrapper validates factor/session
inputs; Alpha evaluation reuses its already validated typed inputs through the
same private numerical kernel. Append retains old pairs, computes only new
target dates and derives the summary once from the merged primitives. Economic
label consistency and overlap checks may share one per-date preparation, but
every constituent still requires the same typed/unique/date/maturity checks.

`evaluation_sample(frames, start=..., end=..., minimum_history_years=None)` owns neutral effective-sample metadata. Saved signal coverage defines the first finite signal and observation calendar; fallback finite economic observations respect maturity and count zero returns. Exclude only leading unavailable history, never trim by return direction or erase internal gaps. Minimum calendar years is an explicit caller policy. No available signal means no economic sample, not a fabricated flat path.

`return_metrics`/`return_statistics` expose empirical daily `expected_shortfall_95`: negative mean of the worst ceil(5% * finite sample size) observations, requiring at least 20. Keep undefined values null with reasons. Saved research aggregation exposes annual/recent performance, annual IC and coverage summaries through the same formulas.


An explicit `minimum_signal_coverage` uses finite signals / full supplied Universe
and an inclusive first qualifying date. Preserve complete leading coverage to
resolve the gate; no qualifying coverage cannot fall back to returns. Pass the
resolved sample to derived windows instead of re-gating a sliced interval.
Fresh diagnostic restarts are caller-owned calendar/input choices, with no old
state. Risk's optional original `factor_catalogue` preserves first_seen metadata
under a late start without pre-start observations or warm-up credit.


Public `pair_correlation` owns incremental saved-input return/daily cross-section
statistics. Match finite coordinates before rank ties, retain unaffected daily
summaries and remove replaced/deleted days. Applications prove complete holdings
sessions before requesting absent-key zero fill. Account/holdings date evidence
is captured on publication or explicit original-bound maintenance, never browsing.
