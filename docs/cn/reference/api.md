# API

## `compose_prediction`

```python
compose_prediction(
    alpha_values,
    composer,
    calendar,
    alpha_policy,
    *,
    execution_policy="next_open",
    prices=None,
)
```

`alpha_values` 用稳定 alias 映射到普通 core `Panel`；结果是终端
`PredictionPanel`。standardization 必须为 `zscore` 或 `percentile_rank`。
IC weighted、OLS 与 GLS 必须提供价格，以构造无前视的 execution-to-next-
execution 标签。

## `run_prediction_backtest`

```python
run_prediction_backtest(
    prediction,
    prices,
    calendar,
    *,
    weight_policy,
    execution_policy="next_open",
    weight_inputs=None,
    config,
    execution_availability=None,
    slippage_rates=None,
)
```

公开边界严格要求 `PredictionPanel`。函数依次应用日期、执行和组合政策，再进入
私有 weights 引擎。裸 DataFrame、普通 `Panel` 和直接 weights 会被拒绝。

## `run_prediction_evaluation`

```python
run_prediction_evaluation(scheduled_signal, prices, *, config, ...)
```

输入必须为 `ScheduledPrediction`。forward return 从本次 execution price 到
下一次 Signal execution price；结果包含 Spearman/Pearson IC、分位数、
spread、TOP N、lag、IC decay、基准和覆盖度。

## 数据边界

Signal 使用 core `PredictionPanel(time, asset_id, value)`；价格仍是包含 `time`、
`asset_id`、`price` 的 long-form Polars 数据。WeightPolicy 产生普通 weights
`Panel`，但 weights 不是公开回测入口。
