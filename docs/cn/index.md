# bagelquant-bt 文档

`bagelquant-bt` 将 AlphaValue Panel 组合成强类型 Prediction，执行研究诊断、Weight Policy
与账户回测。它不负责检索市场数据。

推荐流程为：

```text
AlphaValue Panel -> PredictionComposer -> PredictionPanel -> policies -> result
```

公开预测回测只接受 Core `PredictionPanel`；价格使用按 `time`、`asset_id` 键控的 long-form
Polars 数据。整股账户引擎是独立边界，接收 target weights 和 provider-neutral 市场输入。

## 主要入口

```python
from bagelquant_bt import compose_prediction, run_prediction_backtest
from bagelquant_bt import run_account_backtest
```

## 文档

- [概念](concepts.md)
- [架构](architecture.md)
- [快速开始](quick-start.md)
- [性能说明](performance.md)
- [API](reference/api.md)
- [公开 API](reference/public-api.md)
- [交易成本](reference/transaction-costs.md)
- [整股账户回测](reference/account-backtest.md)
- [Signal 评估](reference/factor-evaluation.md)
- [内部实现](reference/internals.md)
