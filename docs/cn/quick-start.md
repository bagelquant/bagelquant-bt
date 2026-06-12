# 快速开始

`bagelquant-bt` 评估研究输出。它不负责读取数据，也不负责生成因子信号。输入必须是数值型 pandas DataFrame。

## 安装

```bash
uv add bagelquant-bt
```

## 权重回测

当 signal frame 已经是组合权重时，使用 `kind="weights"`。行是日期，列是资产，值是目标权重。

```python
from bagelquant_bt import BacktestConfig, run_backtest

result = run_backtest(
    weights,
    prices,
    kind="weights",
    config=BacktestConfig(initial_capital=1_000_000),
)

result.summary
result.net_cumulative_returns
```

## 因子评估

当第一个 frame 是截面因子分数时，使用 `kind="factor"`。包会计算 forward returns、IC、分位数组合收益和 top-N 回测。

```python
from bagelquant_bt import BacktestConfig, run_backtest

result = run_backtest(
    factor_scores,
    prices,
    kind="factor",
    config=BacktestConfig(
        initial_capital=1_000_000,
        quantiles=5,
        top_n=50,
    ),
)

result.ic_mean
result.top_minus_bottom
```

## 交易成本

```python
from bagelquant_bt import BacktestConfig, TransactionCostConfig

config = BacktestConfig(
    initial_capital=1_000_000,
    transaction_cost=TransactionCostConfig(rate=0.00015, min_fee=5.0),
)
```

最小费用需要 `initial_capital`，因为引擎需要把权重换手转换为交易名义金额。

