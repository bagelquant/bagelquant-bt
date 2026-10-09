# 计算记录

普通读取信任已登记不可变元数据，并保留数值类型检查，不重新计算字节或内容哈希。显式审计检查原始完整性，并将当前版本派生索引与原始证据对照。新输出仍保留规范内容哈希。

Owner 公共 API：`BTStore.describe_reference / index_plan / build_index / verify`。索引维护由冻结计划显式执行，支持取消；普通打开不回填历史，也不改写旧 receipt 或 manifest。缺失、部分或版本不匹配的派生索引不提供 selection/interval 证明；可用的权威元数据精确命中和覆盖父记录快捷证明仍有效。资源预算不改变数值身份。
