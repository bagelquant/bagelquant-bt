# Saved evaluation and statistical evidence

Evaluate saved Core numeric/prediction and weights Nodes or frozen execution
plans. Never build/train upstream, fetch providers or submit work during passive
reads. One pure formula owns each statistic; portfolio mechanics supply paths
and reusable evaluation functions consume explicit saved primitives.

Caller-aligned one-session returns require time, asset_id, forward_return,
interval_start, interval_end and available_date. Time is an alignment label;
BT inserts no lag or price convention. Economic intervals are explicit,
nonoverlapping and sequential; availability is at/after the economic end.
Unmatured labels are unavailable, not zero. Mature missing constituents retain
fixed Book/Spread weight with visible coverage; IC and complete quantile rules
remain strict. Calendar gaps must not compress economic windows.

Alpha/Prediction share evaluation: Pearson/rank IC and IR, cross-sectional and
window returns, gross-one centered-rank Book/Spread, deterministic quantiles,
rank persistence, horizons/buckets, lag/lead-lag and rolling IC/inference.
Default annualization is 252 and quantiles 10. Workbench explicitly supplies
240 and its China label adapter (decision t -> adjusted t+1 to t+2 open).
Book/Spread paths use normalized virtual NAV with proportional full-L1 costs.

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
