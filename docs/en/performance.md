# Performance Notes

The benchmark runner separates data construction from evaluation, runs each
measurement in an isolated subprocess, and reports median compute time and
maximum RSS:

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

| Case | Median data | Median compute | Maximum RSS |
| --- | ---: | ---: | ---: |
| `dense-factor` | 0.024s | 0.819s | 484 MB |
| `monthly-factor` | 1.062s | 14.513s | 2,582 MB |
| `constrained-factor` | 1.151s | 18.211s | 2,908 MB |
| `portfolio-path` | 0.283s | 6.523s total | 734 MB |

A read-only production-shape
sample with 92,125 monthly factor observations, 2,003 assets, 2,534 sessions,
and 29,807 execution-rule rows improved from about `44.0s` to about `10–11s`.
Its observed peak RSS fell from about `2.46 GB` to about `2.17 GB`. This clears
the compute-time target, but not the original `1.72 GB` memory target; the
returned full spread target/executed-weight frames and caller-retained source
frames set the remaining floor.

The portfolio-path case also reports the two paths separately. Its three-run
medians were `0.844s` for one continuous calculation and `5.679s` for 126
checkpointed segments, with `734 MB` maximum RSS.

The main optimizations are:

- one streaming Polars plan builds the valuation grid and forward returns;
- valuation prices are retained lazily and materialized only when requested;
- quantile memberships are evaluated in one market scan, with sparse
  corrections for blocked orders;
- lag families reuse filtered market frames and pre-sorted execution keys;
- fixed-horizon IC decay maps every lag in one grouped plan;
- execution restrictions scan sparse state-change and rule events using
  integer-indexed NumPy state;
- transaction costs and net value use preallocated arrays, while turnover and
  fee inputs retain only non-zero trading events;
- full results publish the already-resolved weight frame without a second
  daily-grid join.

All optimizations preserve the public result schemas and execution semantics.
Continuous numerical values are regression-tested at `rtol=1e-12` and
`atol=1e-12`; dates, assets, event counts, blocked trades, and checkpoint state
remain exact.

The benchmark measures computation only. Persisting 126 monthly chunks,
reloading artifacts, and calculating materialization hashes in `investments`
can still dominate an end-to-end production run and should be optimized in
that application rather than in this package.
