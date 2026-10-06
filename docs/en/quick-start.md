# Quick start

Inputs are saved Core Nodes and explicitly aligned return rows; no upstream
calculation or provider access occurs during evaluation.

```python
from datetime import date
import polars as pl
from bagelquant_core import Domain, Node
from bagelquant_bt import build_alpha_weights, evaluate_weights

times = [date(2024, 1, 2), date(2024, 1, 3)]
domain = Domain(calendar=times, universe=["a", "b"])
alpha = Node.from_domain(
    pl.DataFrame({"time": [times[0], times[0], times[1], times[1]],
                  "asset_id": ["a", "b", "a", "b"], "value": [1., 2., 2., 1.]}),
    domain, value_type="numeric", name="saved_alpha",
)
labels = pl.DataFrame({
    "time": [times[0], times[0], times[1], times[1]],
    "asset_id": ["a", "b", "a", "b"], "forward_return": [0.01, 0.02, 0.01, -0.01],
    "interval_start": [times[0], times[0], times[1], times[1]],
    "interval_end": [times[1], times[1], date(2024, 1, 4), date(2024, 1, 4)],
    "available_date": [times[1], times[1], date(2024, 1, 4), date(2024, 1, 4)],
})
weights = build_alpha_weights(alpha, method="book")
result = evaluate_weights(weights, labels, components=("returns",),
                          available_date=date(2024, 1, 4))
returns = result.tables["returns"]
```

`evaluate_alpha(alpha, labels, ...)` uses the same labels for saved signal
statistics. `evaluate_execution(plans, prices, initial_capital=...)` instead
consumes stable-ID signed integer shares and explicit execution/valuation prices.
See [public contracts](reference/public-api.md) and [costs](reference/transaction-costs.md).
