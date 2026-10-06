# 架构与结果权威

portfolio mechanics 负责权重构建/漂移、显式调度、冻结股数和账户状态；通用
评价函数消费这些结果计算 IC、收益、风险、推断和比较。只有一条执行账户循环，
account.py 仅保留可达的私有状态/成交/结算/企业行动 primitives。

BT 只依赖 Core，不导入 Data/Workbench，也不访问 provider/应用内部状态。
BTStore(meta_path, artifact_path) 独立拥有 SQLite 元数据、不可变 Parquet、
结果身份、receipt、查询、校验、失效、恢复和清理。调用 begin、publish_shared、
publish_chapter、read_reference/read_chapter、history、invalidate、verify、
recover、cleanup_plan/cleanup 等公开 API。失败保留原有有效 receipt，不自动迁移
不兼容数据库。Workbench 保存应用关系与 backend 引用，不复制数值或证明权威。

执行 checkpoint 保存实际/理论账本、FIFO/权益证据、未来及重试计划。续算验证
规范化输入前缀；merge_execution_tables 合并历史；历史窗口使用截止日期的
lot/mark snapshot 和成熟事件，由 summarize_transaction_pnl 聚合，不回放账户。

Workbench 拥有全局调度、资源策略、中国市场声明、研究/治理与 GUI。
BT 接受明确的本地 workers/预算，不探测机器。真实服务/数据切换需另行授权。
