# API

## `run_backtest`

```python
run_backtest(signal, prices, *, kind, config=None)
```

根据 `kind` 分发到对应评估路径：

- `kind="weights"` 调用 `run_weight_backtest`
- `kind="factor"` 调用 `run_factor_evaluation`

由于最低交易费用需要 `initial_capital`，实际使用时需要提供 `config`。

## `run_weight_backtest`

```python
run_weight_backtest(weights, prices, *, config)
```

把 DataFrame 解释为组合权重并执行回测，返回 `BacktestResult`。

重要字段包括 `weights`、`asset_returns`、`gross_returns`、`net_returns`、累计收益、组合价值、换手、交易成本和 `summary`。

## `run_factor_evaluation`

```python
run_factor_evaluation(factor, prices, *, config)
```

把 DataFrame 解释为因子分数并执行评估，返回 `FactorEvaluationResult`。

重要字段包括 `factor`、`forward_returns`、`ic`、`ic_mean`、`ic_std`、`icir`、分位数组合收益、top-minus-bottom、TOP N 权重和 TOP N 回测结果。

## Config

```python
BacktestConfig(
    initial_capital=1_000_000,
    transaction_cost=TransactionCostConfig(rate=0.00015, min_fee=5.0),
    annualization=252,
    ic_method="spearman",
    quantiles=5,
    top_n=50,
)
```

`initial_capital` 必须为正数。`ic_method` 可以是 `"spearman"` 或 `"pearson"`。

## DataFrame 边界

第一个参数必须是数值型 `pandas.DataFrame`。行是日期，列是资产，值是权重或因子分数。
