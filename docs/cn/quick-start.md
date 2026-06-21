# 快速开始

`bagelquant-bt` 评估研究输出。它不负责读取数据，也不负责生成因子信号。输入必须是数值型 long-form Polars DataFrame。

## 安装

```bash
uv add bagelquant-bt
```

## 权重回测

当 signal frame 已经是组合权重时，使用 `kind="weights"`。权重使用 `time`、`asset_id`、`weight` 列；价格使用 `time`、`asset_id`、`price` 列。

```python
import polars as pl

from bagelquant_bt import BacktestConfig, run_backtest

prices = pl.DataFrame(
    {
        "time": ["2024-01-02", "2024-01-03"],
        "asset_id": ["AAA", "AAA"],
        "price": [100.0, 102.0],
    }
)
weights = pl.DataFrame(
    {"time": ["2024-01-02"], "asset_id": ["AAA"], "weight": [1.0]}
)

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

当第一个 frame 是截面因子分数时，使用 `kind="factor"`。因子输入使用 `time`、`asset_id`、`factor` 列。包会计算 forward returns、IC、分位数组合收益和 top-N 回测。

```python
from bagelquant_bt import BacktestConfig, run_backtest

factor = pl.DataFrame(
    {"time": ["2024-01-02"], "asset_id": ["AAA"], "factor": [1.5]}
)

result = run_backtest(
    factor,
    prices,
    kind="factor",
    config=BacktestConfig(
        initial_capital=1_000_000,
        quantiles=5,
        top_n=50,
    ),
)

result.ic_mean
result.spread_returns
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
