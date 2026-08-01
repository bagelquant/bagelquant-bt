# 公开 API

稳定 API 从 `bagelquant_bt` 导出。0.2 只接受强类型 Signal；普通 `Panel`、
裸 DataFrame 和直接 weights 均不能进入公开回测入口。

## 入口函数

```python
from bagelquant_bt import compose_signal, run_signal_backtest, run_signal_evaluation
```

- `compose_signal(...) -> SignalPanel`：按 `SignalDatePolicy` 的 cadence
  执行 `SignalComposer`。监督式 composer 使用本次 execution 到下一次
  execution 的收益，并检查标签可用时间。
- `run_signal_backtest(...)`：依次应用 `SignalDatePolicy`、
  `ExecutionPolicy`、`PortfolioPolicy` 和内部 weights 引擎。
- `run_signal_evaluation(scheduled_signal, prices, ...)`：计算 IC、分位数、
  lag、IC decay 和 Signal 驱动的组合结果。

`PortfolioPolicy` 接收 `ScheduledSignal`，返回
`PortfolioBuild(weights: Panel, skipped: DataFrame)`。`OrderSizingPolicy` 与
`OrderPlan` 只预留 weights 到订单的边界，本版本不实现手数计算。

`BacktestResult` 包含权重、收益、净值、换手、成本、执行阻塞和覆盖度。
`SignalEvaluationResult` 包含 Signal、execution-to-execution forward returns、
Pearson/Spearman IC、分位数、spread、TOP N、lag、IC decay 与基准结果。

## 异常

- `BagelQuantBacktestError`：包级基础异常。
- `BacktestConfigError`：配置无效。
- `InputValidationError`：市场数据无效或不兼容。
