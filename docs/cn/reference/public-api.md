# 公开 API — BT 0.12

`alpha_update_plan(previous, *, calendar, available_date, affected_intervals=(),
preceding=(), ..., input_proof=None)` 仅依据元数据规划追加和受影响后缀并选择原始前置
checkpoint；无安全状态时返回明确恢复计划，通过同一 evaluate_alpha 引擎继续执行。

`SharedInputEvidence` 与 `BTStore.maintain_shared_inputs(owner_id, ...)` 显式维护
旧 signal 日原语的同边界输入绑定。提供原/新 input identity、原 receipt/request、
settings/kernel、read windows、五类完整 source-owner typed 投影和报告引用/SHA256。
BT 只发布登记原文件引用，读取及发布时验证原 authority；不增加 groups，不授予
append/native causal state。详见 [Alpha 评估](factor-evaluation.md)。

Alpha 续算提供 `previous` 与三边界 `input_proof`，`EvaluationResult.state`
保留各路径 checkpoint 和前缀策略。证明合同和当前性能限制见
[Alpha 评估](factor-evaluation.md)。`BTStore.publish_table`、`compose_reference`
与 `publish_continuation` 在同一不可变 authority 下复用原表并写变化日尾部。
组合通过 references-only 发布，原始字节审计保持显式。

原生账户的 input_evidence/prefix_evidence 使用各消费 channel 的完整
SharedInputEvidence。validate_execution_continuation 在任何数值输入读取前校验原
checkpoint 元数据；版本化账户绑定、累计状态与旧 signal 维护分别有自己的权威证据。
BTStore.execution_references 保存新增账户表/小 checkpoint，并保留历史引用；
read_reference(filters=...) 用已登记坐标剪枝、将 eq/ge/le 谓词推入 Parquet。
read_reference(coordinates=typed_keys) 精确选择 Date/String/Integer 多列组合；
空 keys 不读数值文件。原生索引包含 checkpoint_date、available_date；旧坐标
证据缺失时显式 maintain_reference，原 bytes 与 receipt 保留。
channel 范围与失败处理见 [股数账户](account-backtest.md)。

- evaluate_alpha 接受已保存 numeric/prediction Node；evaluate_weights 接受
  weights Node。收益需 time、asset_id、forward_return、interval_start、
  interval_end、available_date；经济区间顺序明确且不重叠，无隐式 lag/价格规则。
- build_alpha_weights(method="book"/"spread") 提供 gross-one/net-zero 权重；
  每 N 日调仓需要显式 calendar/anchor，并取最新因果整体截面。
- evaluate_execution 接受唯一 plan_id、execution_date、asset_id、signed integer
  quantity、execution_price/valuation_price（或显式 open/close）与 initial_capital。
  可选的订单 reference_price 优先于 market reference_price，否则理论账本使用
  调用者的 execution price。
- run_execution_from_weights 使用实际决策日收盘状态形成冻结 delta；需要显式
  calendar/session_lag，decisions 的 time/status 为 rebalance/hold/unavailable。
  reference_price_mode 默认 execution；显式 decision 冻结决策收盘参考价，
  Workbench 选择 decision。
- save_checkpoint/load_checkpoint、merge_execution_tables 和
  summarize_transaction_pnl(frames, through=end) 提供续算与历史读取。
  新发现的历史 record-date 行动需核实实际/理论 position、session 与 lot 历史。
- ExecutionConfig 声明成本/lots/settlement/retry；ExecutionCostRule 需稳定
  ID/version/parameters 和纯 quote，返回 ExecutionCostQuote。
- return_statistics 与风险/统计函数复用通用公式。默认 annualization=252、
  quantiles=10；Workbench 明确传入 240 和中国市场语义。
- BTStore/evaluation_identity 拥有缓存/receipt；BTExecutionOptions 与
  run_evaluation_batch 提供显式本地并行，不负责全局资源分配。

详见[交易](account-backtest.md)、[成本](transaction-costs.md)、[架构](../architecture.md)。

`BTStore(meta_path, artifact_path).inspect()` 为公共只读初始化检查，返回
`uninitialized`、`ready` 或 `incompatible` 及版本/原因。检查不创建目录或数据库，
并验证版本、必需表及必需列；版本正确但列不完整的 schema 仍不可用。
检查复用 Core 的公共元数据快照原语，包含已提交 WAL；SQLite 只打开私有系统
临时副本，检查后删除，不改变原目录、数据库或 WAL/SHM。活动/hot rollback
journal 会拒绝检查，零头失效的 PERSIST journal 可读；不恢复、不覆盖历史
证据；空存储只能通过显式 `initialize()` 创建。

`evaluate_alpha` 还接受 `options: BTExecutionOptions | None` 和
`progress: Callable[[int, int | None], None] | None`，在调用方预算内并行 horizon。
`EvaluationResult.execution` 的执行估计独立于 tables/metrics，不得加入评估身份。

`BTStore.inspect(runtime=True)` 为明确的活动服务就绪检查，通过 Core 的只读事务
协调读取已提交 SQLite 状态，不复制活动元数据，不初始化/恢复，不接受未提交修改。
正常 WAL 读取协调可更新 SHM 读标记；持久化数据保持不变。默认 `runtime=False`
继续用于 setup/安装的严格离线检查，保留源文件与 sidecar 字节。

`evaluation_sample` 与 `SAMPLE_POLICY` 提供保存有效样本契约；[研究统计](research-statistics.md)定义日期、缺口与经验 ES95。
