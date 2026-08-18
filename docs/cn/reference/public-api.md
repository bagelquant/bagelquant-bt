# 公开 API

稳定 API 从 `bagelquant_bt` 导出。0.2 只接受强类型 Signal；普通 `Panel`、
裸 DataFrame 和直接 weights 均不能进入公开回测入口。

## 入口函数

```python
from bagelquant_bt import compose_prediction, run_prediction_backtest, run_prediction_evaluation
```

- `compose_prediction(...) -> PredictionPanel`：按 `AlphaPolicy` 的 cadence
  执行 `PredictionComposer`。监督式 composer 使用本次 execution 到下一次
  execution 的收益，并检查标签可用时间。
- `run_prediction_backtest(...)`：依次应用 `AlphaPolicy`、
  `ExecutionPolicy`、`WeightPolicy` 和内部 weights 引擎。
- `run_prediction_evaluation(scheduled_signal, prices, ...)`：计算 IC、分位数、
  lag、IC decay 和 Signal 驱动的组合结果。
- `quantile_rank_information_coefficients(quantile_returns)`：从 q1 到 qN 的 gross
  组收益生成逐期单调性 rank IC。

候选预测验证由 `score_ic_validation`、`select_top_n_stable`、
`top_n_monthly_performance` 和 `score_top_n_performance` 提供。IC 未定义或 prediction
为常数的月份不会按零计分；有效月份不足时候选无效。Top-N 对 cutoff tie 使用 prediction
降序、asset ID 升序，稳定选取恰好 N 只并返回 cutoff/tie 审计。换手正则目标明确定义为
`net_sharpe - lambda * average_turnover`。
`top_n_monthly_performance` 可以使用紧凑的比例换手成本，也可以分别接收佣金、逐资产
最低佣金、卖出税、滑点与初始资金。

`WeightPolicy` 接收 `ScheduledPrediction`，返回
`WeightBuild(weights: Panel, skipped: DataFrame)`。手数计算与实盘报单不属于本包边界。

`BacktestConfig.insolvency_action` 默认为 `"raise"`，保持严格失败语义。设为
`"freeze_zero"` 后，资不抵债当日的有效费用封顶为可用财富，同时记录请求费用和未支付
费用，净收益记为 `-100%`；之后 gross/net 收益与交易均冻结为零。return、lag 和
quantile 路径会提供 `is_bankrupt` 与 `bankruptcy_event` 标记。

`BacktestResult` 包含权重、收益、净值、换手、成本、执行阻塞和覆盖度。
`PredictionEvaluationResult` 包含 Signal、execution-to-execution forward returns、
Pearson/Spearman IC、分位数、spread、TOP N、lag、IC decay 与基准结果。

## 异常

- `BagelQuantBacktestError`：包级基础异常。
- `BacktestConfigError`：配置无效。
- `InputValidationError`：市场数据无效或不兼容。
