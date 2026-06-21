# 因子评估

因子评估会把因子 DataFrame 解释为截面分数，分数越高代表越好。

## IC 和 ICIR

对每个日期，`bagelquant-bt` 计算 t 日因子分数与 t 到 t+1 资产收益之间的截面相关性。

因子评估会在 `result.ic` 中同时输出 Pearson 相关和 Spearman 秩相关：

```python
result.ic.select("time", "pearson_ic", "spearman_ic")
```

`result.ic_summary` 包含每种方法的均值、标准差和 ICIR。兼容字段
`ic_mean`、`ic_std` 和 `icir` 使用 Spearman IC。

`icir` 定义为：

```text
mean(IC) / standard_deviation(IC)
```

## 分位数组合收益

每天按因子分数从高到低排序资产，并切分为若干分位数组：`q1` 为最高分数组，`qN` 为最低分数组。每个分位数组合收益是组内资产前向收益的等权平均。

spread 为：

```text
q1 收益 - qN 收益
```

## TOP N 回测

TOP N 回测会把因子分数转换成长-only 等权组合：

```text
每天前 N 个资产 -> 每个资产 1 / N 权重
```

生成的权重表会进入与普通权重回测相同的引擎，包括交易成本计算。

## Spread 和滞后分析

因子评估还会构造 spread 组合：做多 `q1`、做空 `qN`，并通过同一个含交易成本的回测引擎计算结果。

`lag_analysis` 会评估 TOP N 和 spread 组合在因子信号滞后 0、1、2、3、4、5、10、20、30、60 个观测后的累计收益和 Sharpe。

`lag_returns` 包含相同组合和滞后下的 gross/net 累计收益时间序列。

`ic_decay` 会在相同滞后上报告 Pearson 和 Spearman 的平均 IC，并以 IC decay 折线图展示。
