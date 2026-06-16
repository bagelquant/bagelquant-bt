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

把 long-form Polars DataFrame 解释为组合权重并执行回测，返回 `BacktestResult`。

重要字段包括 `weights`、`asset_returns`、`gross_returns`、`net_returns`、累计收益、组合价值、换手、交易成本、`summary` 和 `performance`。

## `run_factor_evaluation`

```python
run_factor_evaluation(factor, prices, *, config)
```

把 long-form Polars DataFrame 解释为因子分数并执行评估，返回 `FactorEvaluationResult`。

重要字段包括 `factor`、`forward_returns`、`ic`、`ic_summary`、`ic_mean`、`ic_std`、`icir`、分位数组合收益、top-minus-bottom、TOP N 权重、TOP N 回测结果、long-short 回测结果、`lag_analysis`、`lag_returns` 和 `ic_decay`。

## `summary_report`

```python
summary_report(
    result,
    *,
    output_path=None,
    missing_price_keys_output_path=None,
    title=None,
    annualization=252,
)
```

为 `BacktestResult` 或 `FactorEvaluationResult` 生成静态 HTML 报告。报告包含精简汇总表格和 Plotly 图表；如果传入 `output_path`，会写入文件并同时返回 HTML 字符串。

因子报告会分为 IC and ICIR、TOP N、spread performance 和 quantile performance 四个部分，每个部分先展示精简表格，再展示图表。

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

`initial_capital` 必须为正数。`ic_method` 仍可传入以保持兼容；因子评估现在会同时输出 Spearman 和 Pearson IC。

## DataFrame 边界

第一个参数必须是数值型 `polars.DataFrame`。权重需要 `time`、`asset_id`、`weight` 列；因子需要 `time`、`asset_id`、`factor` 列；价格需要 `time`、`asset_id`、`price` 列。
