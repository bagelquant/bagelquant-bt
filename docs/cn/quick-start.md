# 快速开始

`bagelquant-bt` 将 AlphaValue Node 组合成强类型 Signal，回测只能通过该
Signal 契约进入。

```python
from bagelquant_core import IdentityPredictionOperator, Node
from bagelquant_bt import (
    BacktestConfig,
    MissingSnapshotAction,
    EvaluationAnchor,
    AlphaPolicy,
    compose_prediction,
    run_prediction_backtest,
)

alpha_value = Node.from_domain(alpha_frame, domain, name="quality")
policy = AlphaPolicy(
    id="month_end",
    frequency="monthly",
    anchor=EvaluationAnchor.LAST_TRADING_DAY,
    missing_snapshot=MissingSnapshotAction.PREVIOUS_IN_PERIOD,
)
signal = compose_prediction(
    {"quality": alpha_value},
    IdentityPredictionOperator(),
    calendar,
    policy,
    standardize_policy="z_score",
)
result = run_prediction_backtest(
    signal,
    prices,
    calendar,
    policy,
    config=BacktestConfig(initial_capital=1_000_000, top_n=50),
)
```

使用 `ICWeightedPredictionOperator`、`ICWeightedDecayPredictionOperator`、
`OLSPredictionOperator` 或 `GLSPredictionOperator` 时，还需向 `compose_prediction`
提供 `prices`。rolling window 与 half-life 按 AlphaPolicy 的交易期计数，不按日频行计数。普通 Node、裸 DataFrame
与直接 weights 均不能传给 `run_prediction_backtest`。

`AlphaPolicy` 只负责选择评估观测；横截面标准化由独立的 `StandardizePolicy` 负责。
规范 registry ID 为 `"none"`、`"z_score"` 和 `"percentile_rank"`。
