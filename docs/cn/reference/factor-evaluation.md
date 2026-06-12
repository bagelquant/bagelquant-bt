# 因子评估

因子评估会把因子 DataFrame 解释为截面分数，分数越高代表越好。

## IC 和 ICIR

对每个日期，`bagelquant-bt` 计算 t 日因子分数与 t 到 t+1 资产收益之间的截面相关性。

默认 IC 方法是 Spearman 秩相关：

```python
BacktestConfig(initial_capital=1_000_000, ic_method="spearman")
```

`icir` 定义为：

```text
mean(IC) / standard_deviation(IC)
```

## 分位数组合收益

每天按因子分数排序资产，并切分为若干分位数组。每个分位数组合收益是组内资产前向收益的等权平均。

top-minus-bottom spread 为：

```text
最高分位收益 - 最低分位收益
```

## TOP N 回测

TOP N 回测会把因子分数转换成长-only 等权组合：

```text
每天前 N 个资产 -> 每个资产 1 / N 权重
```

生成的权重表会进入与普通权重回测相同的引擎，包括交易成本计算。
