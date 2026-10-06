# bagelquant-bt 文档

BT 提供 Alpha/Prediction、组合权重和具体股数交易三种评估，负责金融计算、
通用统计及结果缓存。Core 拥有上游数值和模型，Data 拥有数据/PIT，Workbench
调用公共 API，保留中国市场语义、应用/治理和全局调度。

- [快速开始](quick-start.md)
- [架构与缓存](architecture.md)
- [Workers 与资源](performance.md)
- [公开 API](reference/public-api.md)
- [Alpha 评估](reference/factor-evaluation.md)
- [交易与 FIFO](reference/account-backtest.md)
- [成本](reference/transaction-costs.md)
- [统计推断](reference/research-statistics.md)

旧 composition/policy/account runner 和 Legacy 运行入口已删除，不提供兼容层。
真实数据库与服务切换仍为需单独授权的第 6 阶段。
