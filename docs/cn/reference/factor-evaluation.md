# Alpha / Prediction 评估

evaluate_alpha 消费已保存值与调用方对齐的单期 forward return，共享 IC/rank IC、
ICIR/rank ICIR、autocorrelation/persistence、horizon/bucket、quantile、lag、
lead-lag、rolling IC 和推断。直接信号统计不要求形成账户。

`alpha_update_plan(previous, calendar=..., available_date=...,
affected_intervals=[(start, end)], preceding=[saved_state, ...])` 仅检查小型状态，
返回 initialize/up_to_date/append/repair/blocked、read_start、output_start 和
previous_index（-1 表示当前结果，非负数表示 preceding 中的位置）。修订计划包含
horizon/persistence/lag 的影响并选择同数值策略的最近安全前置结果，可通过 input_proof
检查原 cutoff 的元数据前缀。随后调用同一 `evaluate_alpha(previous=selected)` 执行后缀；
无安全状态或影响覆盖全部历史时给出明确恢复计划。保存的 pending_horizon_start 使任意
延迟标签仍保留原始读取范围。

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

`previous: EvaluationResult` 与 `input_proof(first, observation_through,
information_cutoff)` 支持同策略续算。证明必须绑定 typed Alpha、Domain/traces、
calendar 映射以及该信息边界实际消费的全部标签、经济区间和可用日期。观察边界
与信息边界独立；更晚的信息 cutoff 可能消费超出已存 Alpha 终点的标签。
未知/变化前缀、截止日倒退、缺失日原语或路径状态须显式维护/规划受影响区间，
不能静默重放完成历史。

`state` 保存策略、边界、输入前缀和独立 Book/Tail/lag/quantile checkpoint。
成熟 horizon 日原语复用，仅继续成熟尾部、新 persistence 目标及各路径实际
checkpoint 后的状态。`execution.path_sessions` 记录真正模拟的新增路径日期。
汇总/推断仍消费已存日原语；完整 typed 输入准备和滚动汇总仍有性能缺口。

`BTStore.publish_continuation(tables, previous=references)` 保留相同历史前缀，
仅写变化日尾部，但当前仍读取旧 BT typed 表比较，包括完全相同的结果复用；
不能宣传零历史读取。`publish_table/compose_reference` 使用同一不可变存储。
组合按登记 Date/String/Integer 坐标选择原表，保留顺序与精确行数；逻辑 hash
标识组合本身，不等于平坦表 hash。组合使用 `file_references` 发布，不同时传 frame。
缺索引须显式 `index_plan/build_index`；完整 audit 对照原始行检查坐标事实。
发布在与 cleanup 共用的 writer fence 中检查所有原文件仍存在。

`BTStore.maintain_shared_inputs` 显式维护旧 signal 日原语的输入绑定，限原观察及
成熟边界、原数值 settings/kernel。`SharedInputEvidence` 须提供 value、Domain、
price、calendar、label 五类完整 typed 投影，包含实际 source cutoff、selector、
schema、空坐标、null/nonfinite 与 traces，并保留 source-owner 报告引用/SHA256。
BT 检查投影相等及登记原 receipt 链；外部 source 语义由调用方及 source owner
证明。维护只发布原文件引用，不读取或重算数值，不增加 groups/native scope/state。
缺失 signal coverage 可单独首次初始化并保存。变化/未知输入或更晚评估边界仍须
显式规划或维护 causal state；该 API 不授权历史重放。
