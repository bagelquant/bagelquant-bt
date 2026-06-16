# 公开 API

稳定 API 从 `bagelquant_bt` 导出。

## 入口函数

```python
from bagelquant_bt import run_backtest, run_factor_evaluation, run_weight_backtest
```

- `run_backtest(signal, prices, *, kind, config=None)`：根据 `kind` 分发。
- `run_weight_backtest(weights, prices, *, config)`：评估组合权重。
- `run_factor_evaluation(factor, prices, *, config)`：评估因子分数。
- `summary_report(result, *, output_path=None, title=None, annualization=252)`：
  为 `BacktestResult` 或 `FactorEvaluationResult` 生成静态 HTML 报告。

## 配置

```python
from bagelquant_bt import BacktestConfig, TransactionCostConfig

config = BacktestConfig(
    initial_capital=1_000_000,
    transaction_cost=TransactionCostConfig(rate=0.00015, min_fee=5.0),
    annualization=252,
    ic_method="spearman",
    quantiles=5,
    top_n=50,
)
```

- `initial_capital` 必须为正。
- `ic_method` 仍可传入以保持兼容；因子评估会同时输出 Spearman 和 Pearson IC。
- `quantiles` 控制因子分桶数量。
- `top_n` 控制 top-N 因子组合。

## 结果对象

`BacktestResult` 暴露：

- `weights`
- `asset_returns`
- `gross_returns`
- `net_returns`
- `gross_cumulative_returns`
- `net_cumulative_returns`
- `gross_value`
- `net_value`
- `turnover`
- `transaction_costs`
- `summary`
- `performance`

`FactorEvaluationResult` 暴露：

- `factor`
- `forward_returns`
- `ic`
- `ic_summary`
- `ic_mean`
- `ic_std`
- `icir`
- `quantile_returns`
- `quantile_cumulative_returns`
- `top_minus_bottom`
- `top_n_weights`
- `top_n_backtest`
- `long_short_weights`
- `long_short_backtest`
- `lag_analysis`
- `lag_returns`
- `ic_decay`

## 异常

- `BagelQuantBacktestError`：包级基础异常。
- `BacktestConfigError`：配置无效。
- `InputValidationError`：输入 frame 无效或不兼容。
