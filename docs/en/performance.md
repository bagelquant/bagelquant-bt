# Performance Notes

Benchmarks are intentionally lightweight and reproducible:

```bash
uv run python examples/benchmark_efficiency.py
```

Current local baseline after the executable-weight expansion optimization:

```text
elapsed_seconds=1.186
ic_rows=299
lag_analysis_rows=20
lag_returns_rows=5710
```

The previous local baseline was about `2.38s` for the same synthetic factor
evaluation. The improvement comes from reusing prepared market data and using a
vectorized as-of expansion for portfolio weights instead of nested Python loops
over execution dates and assets.
