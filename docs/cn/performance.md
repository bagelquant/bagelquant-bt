# 性能说明

基准脚本会将数据构造与计算分开计时，并在隔离子进程中运行每次测量，报告计算耗时中位数和最大 RSS：

```bash
uv run python examples/benchmark_efficiency.py --case dense-factor --runs 3
uv run python examples/benchmark_efficiency.py --case all --runs 3
```

可用场景包括：

- `dense-factor`：250 个资产、300 个日频观测；
- `monthly-factor`：2,000 个资产、2,520 个交易日及月频信号；
- `constrained-factor`：生产形状月频信号及稀疏交易限制；
- `portfolio-path`：一次连续计算，以及等价的 126 段 checkpoint 恢复计算。

2026 年 7 月开发机上的三次合成基准结果：

| 场景 | 数据中位耗时 | 计算中位耗时 | 最大 RSS |
| --- | ---: | ---: | ---: |
| `dense-factor` | 0.024s | 0.819s | 484 MB |
| `monthly-factor` | 1.062s | 14.513s | 2,582 MB |
| `constrained-factor` | 1.151s | 18.211s | 2,908 MB |
| `portfolio-path` | 0.283s | 合计 6.523s | 734 MB |

一组只读生产形状样本包含 92,125
条月频因子观测、2,003 个资产、2,534 个交易日和 29,807 条交易规则，
含约束评估从约 `44.0s` 降至约 `10–11s`，峰值 RSS 从约 `2.46 GB`
降至约 `2.17 GB`。计算耗时目标已经达到，但尚未达到原定的 `1.72 GB`
内存目标；完整 spread 的目标/执行权重表及调用方保留的源价格表构成了
当前剩余内存下限。

组合路径场景会分别报告两条路径。三次运行的中位数为：连续单次计算
`0.844s`，126 段 checkpoint 计算 `5.679s`，最大 RSS 为 `734 MB`。

主要优化包括：

- 用一条流式 Polars 管线生成估值网格和 forward return；
- 估值价格按需物化，调用方未请求时不长期保留；
- 分位数组合共用一次市场扫描，blocked order 通过稀疏修正处理；
- lag 家族复用已过滤市场表和排序后的执行键；
- 固定期限 IC decay 在一条带 lag 维度的分组计划中完成；
- 交易限制仅扫描稀疏状态变化和规则事件，并用整数索引 NumPy 数组保存状态；
- 最低手续费和净值路径使用预分配数组，换手和费用输入只保留非零交易事件；
- 完整结果直接发布已解析权重，避免再次连接整张日频网格。

所有优化都保持公开结果 schema 和交易语义不变。连续数值使用
`rtol=1e-12, atol=1e-12` 做回归验证；日期、资产、事件计数、blocked
trade 和 checkpoint 状态要求完全一致。

基准只统计计算。`investments` 中逐月持久化、产物重载和 materialization
hash 仍可能主导 126 段生产链路的总耗时，应在应用层的后续优化中单独处理。
