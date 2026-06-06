# Transaction Costs

The default cost model is:

```python
TransactionCostConfig(rate=0.00015, min_fee=5.0)
```

That is 0.015% per traded asset, with a minimum fee of 5.0 per traded asset.

## Calculation

For each date and asset:

```text
delta_weight = abs(current_weight - previous_weight)
traded_notional = delta_weight * portfolio_value_before_trade
raw_fee = traded_notional * rate
fee = max(raw_fee, min_fee) when traded_notional > 0
```

Daily total fees are divided by portfolio value before trading:

```text
cost_return = total_fee / portfolio_value_before_trade
net_return = gross_return - cost_return
```

The cost-aware portfolio value compounds through time using net returns.

## Result Fields

`BacktestResult.transaction_costs` contains:

- `traded_asset_count`
- `traded_notional`
- `raw_fee`
- `min_fee_adjustment`
- `total_fee`
- `cost_return`

Every backtest includes both gross no-cost and net cost-adjusted results.

