# 快速开始

`bagelquant-bt` 将 AlphaValue Panel 组合成强类型 Signal，回测只能通过该
Signal 契约进入。

```python
from bagelquant_core import IdentitySignalComposer, Panel
from bagelquant_bt import (
    BacktestConfig,
    MissingSnapshotAction,
    SignalAnchor,
    SignalDatePolicy,
    compose_signal,
    run_signal_backtest,
)

alpha_value = Panel.from_domain(alpha_frame, domain, name="quality")
policy = SignalDatePolicy(
    id="month_end",
    frequency="monthly",
    anchor=SignalAnchor.LAST_TRADING_DAY,
    missing_snapshot=MissingSnapshotAction.PREVIOUS_IN_PERIOD,
)
signal = compose_signal(
    {"quality": alpha_value},
    IdentitySignalComposer(),
    calendar,
    policy,
    standardization="zscore",
)
result = run_signal_backtest(
    signal,
    prices,
    calendar,
    policy,
    config=BacktestConfig(initial_capital=1_000_000, top_n=50),
)
```

使用 `ICWeightedSignalComposer`、`OLSSignalComposer` 或
`GLSSignalComposer` 时，还需向 `compose_signal` 提供 `prices`。rolling window
按 SignalDatePolicy 的交易期计数，不按日频行计数。普通 Panel、裸 DataFrame
与直接 weights 均不能传给 `run_signal_backtest`。
