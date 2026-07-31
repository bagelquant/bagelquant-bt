# Performance Notes

The benchmark runner separates data construction from evaluation, runs each
measurement in an isolated subprocess, and reports median compute time, the
data-stage peak, the default result peak, and first weight-materialization cost:

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

| Case | Median data | Data peak | Median compute | Default peak | First weight access |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense-factor` | 0.024s | 135 MB | 0.768s | 496 MB | 0.010s |
| `monthly-factor` | 1.651s | 1,600 MB | 7.446s | 2,009 MB | 1.443s |
| `constrained-factor` | 1.151s | 1,580 MB | 6.557s | 2,095 MB | 1.121s |
| `portfolio-path` | 0.225s | 568 MB | 5.606s total | 727 MB | n/a |

The constrained production-shape synthetic case clears the `7s` second-stage
compute target. Its `2.095 GB` maximum RSS is lower than the earlier `2.46 GB`
real-sample baseline, but does not clear the original `1.72 GB` target. Data
construction alone peaks around `1.58 GB`, leaving a narrow budget for factor
selection, lag state, and result frames. The former read-only real artifact
(92,125 factor observations, 2,003 assets, 2,534 sessions, and 29,807 rule rows)
was no longer present for this run, so the old `44.0s` result is retained only
as historical context and is not presented as a rerun.

The portfolio-path case also reports the two paths separately. Its three-run
medians are `0.716s` for one continuous calculation and `4.890s` for 126
checkpointed segments, with `727 MB` maximum RSS. Both improved rather than
regressed against the previous `0.844s` and `5.679s` medians.

The main optimizations are:

- one streaming Polars plan builds the valuation grid and forward returns;
- valuation prices are retained lazily and materialized only when requested;
- quantile, TOP N, spread, and lag portfolios use sparse holding-state events
  instead of dense snapshot-by-asset target grids;
- lag portfolios share pre-sorted execution keys and batched market-state
  scans; IC decay aggregates each lag immediately to bound rank memory;
- execution restrictions encode rule and retry events once, then scan all
  portfolios using integer-indexed NumPy state;
- transaction costs and net value use preallocated arrays, while turnover and
  fee inputs retain only non-zero trading events;
- full results defer expanded target/executed weights until the corresponding
  field is first accessed; materialization is thread-safe and cached.

All optimizations preserve the public result schemas and execution semantics.
Continuous numerical values are regression-tested at `rtol=1e-12` and
`atol=1e-12`; dates, assets, event counts, blocked trades, and checkpoint state
remain exact.

The benchmark measures computation only. Persisting 126 monthly chunks,
reloading artifacts, and calculating materialization hashes in `investments`
can still dominate an end-to-end production run and should be optimized in
that application rather than in this package.
