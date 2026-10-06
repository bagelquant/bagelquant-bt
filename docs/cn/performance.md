# Workers 与资源

BTExecutionOptions 明确传入 workers 与 Core ResourceLimits。
run_evaluation_batch 限制独立任务在途数量、分配共享预算、保持输入顺序，并在
取消/失败时停止继续接纳。BT 不探测 CPU/RAM，也不选择全局资源策略；worker、
内存、batch 属于执行配置，不改变数值身份。

单个账户因现金、结算、FIFO 和后续真实决策依赖前序成交，保持因果顺序。
输出使用有界列式 buffer，在每个 session/order/planning 点检查取消。
独立场景可共享只读输入，但不得共享现金/持仓/checkpoint。
先查已校验缓存，再准备市场数据。历史窗口只聚合保存的 primitives/快照，
不会重建上游或重新执行账户。测试使用合成输入与独立临时根。
