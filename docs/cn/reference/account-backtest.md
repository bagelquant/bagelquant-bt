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

Fixed-notional 模式通过注入或申请取出现金维持 notional。外部资金流改变 fund units，不改变
单位 NAV。受限提款保持显式；系统禁止负现金和隐含杠杆。Compounding 模式不产生外部流，
使用当前 equity sizing。

`AccountBacktestResult` 保存 target weights/positions、orders、fills、每日持仓、现金、应收、
外部流、pending withdrawal、账户 equity、performance NAV、derived executable weights、
target/implementation/cost drag 和可恢复 checkpoint。
