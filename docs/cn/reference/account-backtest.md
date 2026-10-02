# 整股账户回测

`run_account_backtest` 是独立的确定性日频账户引擎。输入包括 target weights、与 provider
无关的非复权 open/close、公司行动及覆盖、可交易性、lot size、初始现金/持仓和
`AccountBacktestConfig`。

每个交易日从上一 checkpoint 恢复，释放结算和公司行动应收，按 open 估值，按配置处理
fixed-notional 外部流，计算整数目标股数，先执行可卖订单并支付 pending withdrawal，最后
分配可负担买入整手。买入按最大跟踪误差改善排序，使用稳定 `asset_id` 打破平局。订单要么
完整成交，要么明确标记为受限或缩减；最新 target revision 取代旧 pending intent。

引擎不会用复权价格合成股数。Record-date 收盘持仓确定分红权益；ex-date 创建现金和股票
应收；pay-date 释放现金；股票可用日释放红股。每个模拟交易日都必须具有完整公司行动覆盖。

恢复 `run_stateful_account_backtest` 时，除 checkpoint 外，还应通过
`initial_target_position_plans` 传入已冻结的计划。在 checkpoint 之前决定、之后才到执行日的
计划会按原日期和原股数执行，不重复调用其决策回调，也不在执行日开盘重新确定股数。
保存目标的 `evaluate_portfolio_targets` 使用相同规则：日历须包含已知的未来执行日期，
价格仍截止模拟截止日。保存返回的未来计划并在续算时传入；无法安排决策的截断日历
会明确报错。`prepare_account_market_data` 为多个独立压力账户共用已校验的价格及可交易性查询
结构，其中不包含账户状态。

检查点保留零股数坐标和原目标修订日期，使后续退出与待成交订单的审计身份和连续
回测一致。追加公司行为的权益登记日在检查点之前时，必须传入已验证的
`historical_positions` 和 `historical_sessions`，按登记日收盘持仓补齐权益，不使用
当前持仓代替。完整历史账户交易日清单之外的登记日不产生账户权益；所需历史交易日
缺少持仓记录时明确失败。

Fixed-notional 模式通过注入或申请取出现金维持 notional。外部资金流改变 fund units，不改变
单位 NAV。受限提款保持显式；系统禁止负现金和隐含杠杆。Compounding 模式不产生外部流，
使用当前 equity sizing。

`AccountBacktestResult` 保存 target weights/positions、orders、fills、每日持仓、现金、应收、
外部流、pending withdrawal、账户 equity、performance NAV、derived executable weights、
target/implementation/cost drag 和可恢复 checkpoint。

整手分配先最大化股票投入金额，再最小化目标偏差。HiGHS 出现数值求解错误时，只关闭
presolve 并对同一模型重试一次；目标、上下界、资金预算及整数约束均保持不变，不放宽
不可行约束，也不接受未完成求解的结果。取整后的股数仍须满足股票预算和第一阶段的
投入金额下限。
