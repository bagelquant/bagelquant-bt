# 公开 API — BT 0.12

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
