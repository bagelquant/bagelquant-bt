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

`evaluate_alpha(..., options=..., progress=...)` 复用同一个有界池并行独立
horizon 窗口。factor、labels、Book/Spread 只准备一次，各窗口私有结果按声明
顺序合并，再统一做推断。未提供 options 或 factor 不满 1024 行时保持串行。
progress 在调用线程报告 `(已完成窗口数, 总窗口数)`，从零开始；窗口阶段与
汇总之间检查取消。调用方只能选择外层对象并行或内层窗口并行，避免嵌套池。

并发估计预留调用方内存预算的 20% 加共享 frame 的 estimated_size；每窗口
工作区取 64 MiB 与 factor+labels 大小的 8 倍中的较大值，据此减少 workers，
最少为 1。这是保守接纳估计，不是硬内存分配限制。`EvaluationResult.execution`
报告请求/实际 workers 与各字节估计，属于执行证据，不得加入数值 metrics/身份。

章节目录可读取元数据 receipt，避免打开全部周期文件。选择性读取在有限 context 中对每个选中文件只哈希一次，并在解码后重查文件身份；重复读取复用证明。无关周期或 table 的损坏由全量审计或实际消费时发现。评估公式、因果执行与结果身份保持一致。
