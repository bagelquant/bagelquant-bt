# 架构与设计

`bagelquant-bt` 是强类型 Signal 优先的评估包。

```text
AlphaValue Panel + prices + policies
    |
    v
PredictionComposer -> PredictionPanel
    |
    v
schedule -> execution -> portfolio weights
    |
    v
收益、换手、成本、IC、分位数
    |
    v
可视化辅助函数
```

## 设计哲学

- Alpha 定义和市场数据读取位于回测包之外。
- 公开回测边界严格要求 `PredictionPanel`。
- 交易成本必须显式且可复现。
- 返回结构化结果对象，而不是只打印报告。
- 可视化层只消费结果对象。

## 结构

- `inputs`：frame 校验、对齐和数值检查。
- `returns`：资产收益和累计收益工具。
- `costs`：换手和交易成本计算。
- `pipeline`：Signal 组合和严格公开回测编排。
- `signal`：AlphaPolicy 与 ExecutionPolicy。
- `portfolio`：`ScheduledPrediction` 到 weights 的政策。
- `allocation`：通用、确定性的连续目标权重到整数手数仓位分配。
- `engine`：包内 weights 模拟。
- `factor`：IC、分位数和 top-N Signal 评估。
- `performance`：汇总指标。
- `results`：供下游检查的 dataclass。
- `visualization`：绘图辅助函数。

## 数据边界

AlphaValue 和 weights 是普通 core `Panel`；组合后的 prediction 是
`PredictionPanel`；`ScheduledPrediction` 额外保存 schedule 与日期 lineage。价格 frame
保存用于计算收益的数值价格。

包依赖 `bagelquant-core` 的 Panel 与 composer 契约，但不导入
`bagelquant-data` 或 `bagelquant-workbench` 应用代码。
