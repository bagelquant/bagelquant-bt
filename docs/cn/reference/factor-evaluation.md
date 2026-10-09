# Alpha / Prediction 评估

evaluate_alpha 消费已保存值与调用方对齐的单期 forward return，共享 IC/rank IC、
ICIR/rank ICIR、autocorrelation/persistence、horizon/bucket、quantile、lag、
lead-lag、rolling IC 和推断。直接信号统计不要求形成账户。

Book 为居中平均排名并将绝对权重归一到 1；Spread 上/下尾各 +0.5/-0.5。
两者净敞口为 0。build_alpha_weights 提供 weights Node，路径复用权重引擎，
采用虚拟 normalized NAV 与比例 full-L1 成本。

累计窗口 1/5/10/20/40/60/120 sessions 与 bucket 使用显式经济区间/成熟日期。
未成熟标签不可用，不能当作零。已成熟 constituent 缺口保持原权重，贡献零并
报告覆盖率；IC/完整 quantile 严格处理。不得根据未来收益可用性重新选股/归一。
稀疏 calendar 不压缩窗口。调仓用最新因果整体截面，资产顺序/排名 ties 确定。
负 lag 是明确的研究诊断，不可当执行计划。

显式 `options: BTExecutionOptions | None` 与 `progress(完成数, 总数)` 支持
调用方预算内的独立 horizon 窗口并行。默认串行，保持窗口顺序、公式与成熟日期。
`EvaluationResult.execution` 单独提供 worker/内存估计，不改变 tables/metrics
或数值身份。详见[资源](../performance.md)。
