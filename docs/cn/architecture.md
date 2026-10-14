# 架构与结果权威

旧 signal 同边界输入维护使用公开 `BTStore.maintain_shared_inputs`、完整
source-owner `SharedInputEvidence` 和不可变原执行报告。BT 拥有 admission/保留
receipt 检查，外部 source 语义仍由 source owner 证明。保留原数值文件，不构造
native continuation scope/state；缺失 signal coverage 可首次初始化并保存。
append 仍要求明确的 state/proof，见 [Alpha 评估](reference/factor-evaluation.md)。

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

## Receipt 检查与选择性读取

BTStore 拥有不可变章节 receipt 和 table 完整性。普通章节/选中表读取默认
`verify=False`，使用登记描述、行数和索引 schema 检查，不重新计算字节和逻辑哈希。
显式 `verify` 保留完整字节、逻辑内容和索引审计。发布已登记引用不需要解码；
新数值表仍生成规范内容哈希。`index_plan/build_index` 是显式历史索引维护，
打开存储不会回填历史或修改旧 receipt。有限审计上下文退出后证明失效。

## 增量 signal evaluation

`alpha_input_start` 根据保存的路径状态、待成熟 horizon 和所选章节，规划 value
与 price 的最早必要日期。`evaluate_alpha` 接受有界输入与完整不可变前缀证明；
`previous_input_proof` 验证旧章节较窄的消费范围。补章节只初始化缺失的原语族，
保留已有金融路径。缺少续算状态要求显式维护，调用方保留完整日历和数值设置。

滚动 IC 保存旧输出，只计算新日期和各方法/horizon 必要的有效观测 halo；
null/NaN 不占有效观测数。续算必须保持观测设置并提供原有限 IC halo；空更新
保留原输出。有界浮点求和的末位舍入可能不同，新增 rolling mean 按 1e-12
容差对比完整参考，保存的旧前缀保持精确。summary 可聚合保存的原语。`publish_continuation`
保留原索引 table fragments 并发布新尾部；当前组合仍读取原 typed fragments
验证完整表，这一剩余 I/O 与重算历史日频统计应分别计量。
