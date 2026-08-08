# Signal 评估

Signal 评估会把 `ScheduledPrediction` 中的每个 `PredictionPanel` 快照解释为截面
prediction，数值越高代表越正向。

## IC 和 ICIR

对每个信号期，`bagelquant-bt` 计算 Signal 与本次 execution price 到下一次
Signal execution price 收益之间的截面相关性。

Signal 评估会在 `result.ic` 中同时输出 Pearson 相关和 Spearman 秩相关：

```python
result.ic.select("time", "pearson_ic", "spearman_ic")
```

`result.ic_summary` 包含每种方法的均值、标准差和 ICIR。兼容字段
`ic_mean`、`ic_std` 和 `icir` 使用 Spearman IC。

`icir` 定义为：

```text
mean(IC) / standard_deviation(IC) * sqrt(每年 IC 观测数)
```

`BacktestConfig.ic_annualization` 用于指定每年的 IC 观测数；未指定时，默认使用日收益的 `annualization`。这与组合收益指标不同：年化收益率、波动率、Sharpe、滚动指标、基准和滞后 Sharpe 都使用日收益年化数。

## Signal 与 Execution Policy

`AlphaPolicy.select` 从 `PredictionPanel` 选择完整截面，返回带调度、执行日期
lineage 和强类型 signal 的 `ScheduledPrediction`。`month_end` 优先选择月末最后一个
开市交易日；若整张截面不存在，只能回退到同一自然月内最近的此前完整截面，
否则记录 skip。单个资产绝不会从此前日期回填。

`ExecutionPolicy("next_open")` 独立把再平衡日映射到下一开市交易日。`run_prediction_evaluation` 接收已经解析的统一 selection，IC 使用本次 execution date 到下一次 execution date 的收益，组合则在两次执行之间逐日估值；最后一条没有完整后续执行期的信号不进入 IC。

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

Signal 评估还会构造 spread 组合：做多 `q1`、做空 `qN`，并通过同一个含交易成本的回测引擎计算结果。

`lag_analysis` 会评估 TOP N 和 spread 组合在 Signal 滞后 0、1、2、3、4、5、10、20、30、60 个交易日后的累计收益和 Sharpe；月频 Signal 的 Lag 15 是 15 个开市日，不是 15 个按月观测。

`lag_returns` 包含相同组合和滞后下的 gross/net 累计收益时间序列。

`ic_decay` 会在相同交易日滞后上报告 Pearson 和 Spearman 的平均 IC，并以 IC decay 折线图展示。
