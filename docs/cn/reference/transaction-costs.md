# 交易成本

默认成本模型为：

```python
TransactionCostConfig(
    rate=0.00015,
    min_fee=5.0,
    buy_slippage_rate=0.0005,
    stamp_tax_rate=0.0005,
)
```

`rate` 和 `min_fee` 分别表示双边佣金率和每资产每次交易的最低佣金。
滑点同时应用于买入和卖出；印花税仅在卖出时收取。

## 计算方式

每个执行日、每个资产根据带方向的权重变化判断买卖方向：

```text
signed_delta = target_weight - previous_weight
buy_notional = max(signed_delta, 0) * 交易前组合价值
sell_notional = max(-signed_delta, 0) * 交易前组合价值
traded_notional = buy_notional + sell_notional

slippage_fee = traded_notional * 有效滑点率
raw_fee = traded_notional * 佣金率
commission_fee = max(raw_fee, 最低佣金)
stamp_tax_fee = sell_notional * 印花税率
total_fee = slippage_fee + commission_fee + stamp_tax_fee
```

调用方未提供 `slippage_rates` 时使用配置中的统一滑点率。点时滑点表必须包含
`time`、`asset_id` 和 `slippage_rate`，可选 `is_fallback`；每个资产只向后沿用
已经生效的费率，首个生效日前不回填。没有可用记录时使用统一费率，并增加回退交易数。

每日总费用除以交易前组合价值：

```text
cost_return = total_fee / 交易前组合价值
net_return = gross_return - cost_return
```

## 结果字段

`BacktestResult.transaction_costs` 包含交易资产数、滑点回退资产数、总/买入/卖出
名义金额、滑点、原始佣金、最低佣金补差、佣金合计、印花税、总费用和成本收益率。
历史字段 `raw_fee` 与 `min_fee_adjustment` 继续仅表示佣金分项。

每个回测同时保留无成本的 gross 结果和扣除完整成本后的 net 结果。
