# 快速开始

输入为已保存的 Core Node 与调用方对齐的收益表，不会重建上游或读取 provider。
可运行的完整构造例见[英文快速开始](../en/quick-start.md)。

```python
from bagelquant_bt import build_alpha_weights, evaluate_alpha, evaluate_weights
weights = build_alpha_weights(saved_alpha, method="book")
result = evaluate_weights(weights, forward_returns, components=("returns",),
                          annualization=240, available_date=cutoff)
diagnostics = evaluate_alpha(saved_prediction, forward_returns,
                             annualization=240, quantiles=10, available_date=cutoff)
```

收益表必须含 time、asset_id、forward_return、interval_start、interval_end、
available_date；time 为决策/对齐标签，不能代替真实经济区间。BT 不插入默认 lag
或价格规则。`evaluate_execution` 另需显式交易价格、估值价格和初始资金。
