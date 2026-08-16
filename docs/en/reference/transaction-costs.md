# Transaction Costs

The default cost model is:

```python
TransactionCostConfig(
    rate=0.00015,
    min_fee=5.0,
    buy_slippage_rate=0.0005,
    sell_slippage_rate=0.0005,
    stamp_tax_rate=0.0005,
)
```

`rate` and `min_fee` are the two-sided commission settings. Buy and sell
slippage are configured independently. Stamp tax applies only to sells.

## Calculation

For each execution date and asset, the signed weight change determines side:

```text
signed_delta = target_weight - previous_weight
buy_notional = max(signed_delta, 0) * portfolio_value_before_trade
sell_notional = max(-signed_delta, 0) * portfolio_value_before_trade
traded_notional = buy_notional + sell_notional

slippage_fee = traded_notional * effective_slippage_rate
raw_fee = traded_notional * commission_rate
commission_fee = max(raw_fee, min_fee)
stamp_tax_fee = sell_notional * stamp_tax_rate
total_fee = slippage_fee + commission_fee + stamp_tax_fee
```

The effective slippage rate is the configured rate for the trade side unless
the caller supplies an effective-dated `slippage_rates` frame. Its required columns are
`time`, `asset_id`, and `slippage_rate`; optional `is_fallback` marks an
operator-selected fallback. Rates are held forward per asset and are never
backfilled before their first effective date. A missing prior row uses the
configured buy or sell rate and increments the fallback count.

Daily total fees are divided by portfolio value before trading:

```text
cost_return = total_fee / portfolio_value_before_trade
net_return = gross_return - cost_return
```

## Result Fields

`BacktestResult.transaction_costs` contains traded-asset and slippage-fallback
counts, total/buy/sell notional, `slippage_fee`, `raw_fee`,
`min_fee_adjustment`, `commission_fee`, `stamp_tax_fee`, `total_fee`, and
`cost_return`. `raw_fee` and `min_fee_adjustment` are commission-only fields.

Every backtest includes both gross no-cost and net cost-adjusted results.
