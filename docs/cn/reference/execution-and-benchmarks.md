# 成交约束与 Benchmark

## 市场规则显式启用

`bagelquant-bt` 默认不施加任何涨跌停或交易所规则，也不会根据证券代码、
国家或交易所猜测市场。调用方可以向权重、因子或信号评估传入稀疏
`execution_availability` 表：

```text
time, asset_id, can_buy, can_sell, reason
```

没有约束行的资产和日期默认可以买卖。买入增量被阻止时，回测保留现金或
原持仓，并在之后每个价格日重试；新的目标替代尚未完成的旧目标。卖出只有
在 `can_sell=False` 时才被阻止。换手与成本落在实际成交日，审计记录位于
`BacktestResult.execution_blocks`。

## Benchmark 与超额收益

因子结果默认生成 `universe_equal_weight`。调用方通过
`benchmark_universe` 提供每日 point-in-time 成分，也可以通过
`benchmark_returns` 和 `benchmark_coverage` 追加需要额外数据的具名
benchmark。本包本身不加载市值或指数数据。

`FactorEvaluationResult` 包含：

- `benchmark_returns`
- `benchmark_coverage`
- `benchmark_performance`
- `excess_returns`

每个 benchmark 都与 TOP N gross/net 比较，输出每日收益差、每日收益差
复利累计，以及 `TOP N 累计净值 / benchmark 累计净值 - 1` 的相对超额。
