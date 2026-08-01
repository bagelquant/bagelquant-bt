# API

## `compose_signal`

```python
compose_signal(
    alpha_values,
    composer,
    calendar,
    signal_date_policy,
    *,
    standardization="zscore",
    execution_policy="next_open",
    prices=None,
)
```

`alpha_values` 用稳定 alias 映射到普通 core `Panel`；结果是终端
`SignalPanel`。standardization 必须为 `zscore` 或 `percentile_rank`。
IC weighted、OLS 与 GLS 必须提供价格，以构造无前视的 execution-to-next-
execution 标签。

## `run_signal_backtest`

```python
run_signal_backtest(
    signal,
    prices,
    calendar,
    signal_date_policy,
    *,
    execution_policy="next_open",
    portfolio_policy=None,
    portfolio_inputs=None,
    config,
    execution_availability=None,
)
```

公开边界严格要求 `SignalPanel`。函数依次应用日期、执行和组合政策，再进入
私有 weights 引擎。裸 DataFrame、普通 `Panel` 和直接 weights 会被拒绝。

## `run_signal_evaluation`

```python
run_signal_evaluation(scheduled_signal, prices, *, config, ...)
```

输入必须为 `ScheduledSignal`。forward return 从本次 execution price 到
下一次 Signal execution price；结果包含 Spearman/Pearson IC、分位数、
spread、TOP N、lag、IC decay、基准和覆盖度。

## 数据边界

Signal 使用 core `SignalPanel(time, asset_id, value)`；价格仍是包含 `time`、
`asset_id`、`price` 的 long-form Polars 数据。PortfolioPolicy 产生普通 weights
`Panel`，但 weights 不是公开回测入口。
