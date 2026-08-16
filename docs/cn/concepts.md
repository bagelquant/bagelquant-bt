# 概念

## 职责边界

`bagelquant-bt` 使用价格评估强类型 Signal，并应用投资 timing 与 portfolio policy：

- `bagelquant-core` 负责 `Panel`、`PredictionPanel`、信号构建和研究逻辑。
- `bagelquant-data` 负责数据访问和存储。
- `bagelquant-bt` 负责评估、交易成本、汇总结果和图表。

公开回测边界只接受 `PredictionPanel`。普通 `Panel`、裸 Polars `DataFrame` 或直接 weights
都不能传入 `run_prediction_backtest`。

## DataFrame 形状

价格、Signal 和 weights 都使用同一种 long-form 形状：

```text
keys:    time, asset_id
values:  price, value, 或 weight
```

`AlphaPolicy` 选择 observation snapshot，`ExecutionPolicy` 映射执行 session，
`WeightPolicy.build(ScheduledPrediction)` 返回包含普通 weights `Panel` 与 skipped rows 的
`WeightBuild`。`MarketRule` 保持独立。手数计算与实盘报单不属于本包边界。

## 时间约定

包内采用无前视约定：

```text
在 t 日可用的 Signal -> 只能在 information cutoff 之后执行
执行后的组合权重 -> 获得下一价格日期的收盘到收盘收益
```

没有完全匹配 `(time, asset_id)` 价格键的信号或权重行会从执行中删除，并记录在
`missing_price_keys`。最后一个价格日期没有可实现的前向收益，因此可以保留为输入，
但不会贡献已实现收益。

## 对齐

`bagelquant-bt` 会按完全匹配的价格键对齐价格与信号值。必需列中的 null 和 NaN
会在对齐前被删除。

它会拒绝重复的 `(time, asset_id)` 键、非数值输入，以及公开入口中的非
`PredictionPanel` signal 输入。
