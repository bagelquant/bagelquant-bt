# 快速开始

`bagelquant-bt` 将 AlphaValue Panel 组合成强类型 Signal，回测只能通过该
Signal 契约进入。

```python
from bagelquant_core import IdentityPredictionComposer, Panel
from bagelquant_bt import (
    BacktestConfig,
    MissingSnapshotAction,
    EvaluationAnchor,
    AlphaPolicy,
    compose_prediction,
    run_prediction_backtest,
)

alpha_value = Panel.from_domain(alpha_frame, domain, name="quality")
policy = AlphaPolicy(
    id="month_end",
    frequency="monthly",
    anchor=EvaluationAnchor.LAST_TRADING_DAY,
    missing_snapshot=MissingSnapshotAction.PREVIOUS_IN_PERIOD,
)
signal = compose_prediction(
    {"quality": alpha_value},
    IdentityPredictionComposer(),
    calendar,
    policy,
    standardization="zscore",
)
result = run_prediction_backtest(
    signal,
    prices,
    calendar,
    policy,
    config=BacktestConfig(initial_capital=1_000_000, top_n=50),
)
```

使用 `ICWeightedPredictionComposer`、`OLSPredictionComposer` 或
`GLSPredictionComposer` 时，还需向 `compose_prediction` 提供 `prices`。rolling window
按 AlphaPolicy 的交易期计数，不按日频行计数。普通 Panel、裸 DataFrame
与直接 weights 均不能传给 `run_prediction_backtest`。
