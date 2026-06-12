# 交易成本

默认成本模型是：

```python
TransactionCostConfig(rate=0.00015, min_fee=5.0)
```

也就是每个发生交易的资产按成交名义金额收取 0.015%，且每个资产最低费用为 5.0。

## 计算方式

对每个日期和资产：

```text
delta_weight = abs(current_weight - previous_weight)
traded_notional = delta_weight * portfolio_value_before_trade
raw_fee = traded_notional * rate
fee = max(raw_fee, min_fee) when traded_notional > 0
```

每日总费用除以交易前组合价值：

```text
cost_return = total_fee / portfolio_value_before_trade
net_return = gross_return - cost_return
```

带成本组合价值会使用净收益逐期复利。

## 结果字段

`BacktestResult.transaction_costs` 包含交易资产数量、成交名义金额、原始费用、最低费用调整、总费用和成本收益率。

每个回测都会同时包含不含成本的 gross 结果和扣除成本后的 net 结果。
