# 概念

## 职责边界

`bagelquant-bt` 只负责用日频价格评估研究输出。

它不导入 `bagelquant-core` 或 `bagelquant-data`：

- `bagelquant-core` 负责信号构建和研究逻辑。
- `bagelquant-data` 负责数据访问和存储。
- `bagelquant-bt` 负责评估、交易成本、汇总结果和图表。

因此，只要工作流能产出日期乘资产的 `DataFrame`，就可以使用这个回测包。

## DataFrame 形状

价格、权重和因子分数都使用同一种形状：

```text
keys:    time, asset_id
values:  price, weight, 或 factor
```

价格会被解释为收盘价。权重回测中，输入值是组合权重，可以为负。因子评估中，输入值是截面分数，分数越高表示越好。

## 时间约定

包内采用无前视约定：

```text
t 日的权重或因子 -> 使用完全匹配的价格日期
执行后的组合权重 -> 获得下一价格日期的收盘到收盘收益
```

没有完全匹配 `(time, asset_id)` 价格键的信号或权重行会从执行中删除，并记录在
`missing_price_keys`。最后一个价格日期没有可实现的前向收益，因此可以保留为输入，
但不会贡献已实现收益。

## 对齐

`bagelquant-bt` 会按完全匹配的价格键对齐价格与信号值。必需列中的 null 和 NaN
会在对齐前被删除。

它会拒绝重复的 `(time, asset_id)` 键、非数值输入，以及非 DataFrame 的信号输入。
