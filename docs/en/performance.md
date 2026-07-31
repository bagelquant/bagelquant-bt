# Performance Notes

The benchmark runner separates data construction from evaluation, runs each
measurement in an isolated subprocess, and reports median compute time, the
data-stage peak, the default result peak, first weight-materialization cost,
and the peak after materialization:

```bash
uv run python examples/benchmark_efficiency.py --case dense-factor --runs 3
uv run python examples/benchmark_efficiency.py --case all --runs 3
```

The available cases are:

- `dense-factor`: 250 assets by 300 daily observations;
- `monthly-factor`: 2,000 assets by 2,520 sessions with monthly signals;
- `constrained-factor`: the monthly shape plus sparse execution restrictions;
- `portfolio-path`: one continuous path and the equivalent 126 checkpointed
  segments.

Three-run synthetic results on the July 2026 development machine:

| Case | Median data | Data peak | Median compute | Default peak | First weight access | Materialized peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense-factor` | 0.011s | 109 MB | 0.697s | 534 MB | 0.015s | 534 MB |
| `monthly-factor` | 0.523s | 1,136 MB | 2.904s | 1,146 MB | 0.930s | 1,743 MB |
| `constrained-factor` | 0.532s | 1,143 MB | 3.759s | 1,181 MB | 0.927s | 1,756 MB |
| `portfolio-path` | 0.101s | 331 MB | 5.348s total | 522 MB | n/a | n/a |

The third-stage production-shape synthetic cases clear both compute targets:
`2.904s` versus `5.5s` for monthly evaluation and `3.759s` versus `5.0s` for
constrained evaluation. Their default peaks are below `1.19 GB`, comfortably
under the `1.72 GB` target. Expanded public weights still require about
`1.75 GB`, but that memory is incurred only when requested, and first access
remains below the `1.2s` target. The former read-only real artifact (92,125
factor observations, 2,003 assets, 2,534 sessions, and 29,807 rule rows) was no
longer present for this run, so the old `44.0s` result is retained only as
historical context and is not presented as a rerun.

The portfolio-path case also reports the two paths separately. Its three-run
medians are `0.658s` for one continuous calculation and `4.693s` for 126
checkpointed segments, with `522 MB` maximum RSS. Both improved against the
second-stage `0.716s` and `4.890s` medians.

A separate same-process A/B check against the second-stage commit used 450
assets, 2,520 sessions, and 126 monthly TOP-50 snapshots. The three-run median
for an ordinary `run_weight_backtest` fell from `0.320s` to `0.191s`; data
construction was excluded from both measurements.

The main optimizations are:

- observed-price intervals generate zero-return spans, recovery moves, and gap
  evidence without collecting a dense valuation grid;
- valuation prices retain the public schema but are generated and cached only
  on first access;
- a private read-only market context encodes sessions, assets, observed prices,
  returns, execution keys, and rule retry events once per evaluation;
- quantile, TOP N, spread, and lag portfolios use sparse holding-state events
  instead of dense snapshot-by-asset target grids;
- primary, quantile, spread, and lag portfolios share one market-state scan;
  default TOP N/spread states reuse their lag-zero results;
- absent-price target diagnostics are emitted directly while sparse target
  state is prepared instead of rebuilding dense snapshots;
- IC decay ranks each source factor snapshot once and reranks only groups whose
  return sample is incomplete;
- execution restrictions encode rule and retry events once, then scan all
  portfolios using integer-indexed NumPy state;
- transaction costs and net value use preallocated arrays, while turnover and
  fee inputs retain only non-zero trading events;
- full results defer expanded target/executed weights until the corresponding
  field is first accessed; related results share a thread-safe sorted-key cache;
- on-demand signal diagnostics execute only requested quantile, spread, or lag
  families and combine requested families into one scan.

All optimizations preserve the public result schemas and execution semantics.
Continuous numerical values are regression-tested at `rtol=1e-12` and
`atol=1e-12`; dates, assets, event counts, blocked trades, and checkpoint state
remain exact.

The benchmark measures computation only. Persisting 126 monthly chunks,
reloading artifacts, and calculating materialization hashes in `investments`
can still dominate an end-to-end production run and should be optimized in
that application rather than in this package.
