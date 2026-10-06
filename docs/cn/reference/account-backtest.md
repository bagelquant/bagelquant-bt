# 股数交易、FIFO 与续算

evaluate_execution 接受冻结 signed share delta；卖单优先，再按 asset/plan ID
稳定执行买单，不合并独立订单。实际成交遵循含费用的现金、可卖股数/结算、lot
和 block。余量默认过期；显式 retry 只重试剩余股数，新交易日重新计费。

理论账本按给定 reference price 完整执行原股数，费用为零，忽略实际 lot/settlement/
block；但必须现金和持仓可行，否则整个评估报错。理论/实际独立 FIFO，按匹配
数量分摊买卖费用，未平仓费用/浮盈独立。现金分红归属 record-date 原 buy plan；
送股保留原 plan，明确采用零 acquisition basis（原 lot 保留原成本），不隐含税务规则。
每个 session 的 realized/unrealized/income 与 equity 严格对账。

订单的可选 reference_price 优先于 market 行的 reference_price；均未提供时，
理论账本使用调用者提供的 execution price。参考价格及选择规则进入 checkpoint
identity。某个账本没有成交金额时，其 return 为 null，对应的 target_return_reason
或 actual_return_reason 为 no_filled_notional，不填零收益。

权重桥接使用实际 decision-close 状态冻结 delta，execution open 不重新 sizing。
未来 calendar 必须覆盖执行和结算；价格缺口冻结最后观测价，恢复时确认累计变动。
桥接的 reference_price_mode 默认为 execution；显式 decision 把决策日收盘价
随股数一起冻结。Workbench 为目标收益比较明确选择 decision。
checkpoint 保存两类账本、FIFO/权益、零持仓坐标、未来/重试计划和输入前缀。
新发现 old-record 行动需要核实实际/理论 position、session 和 lot 历史。

merge_execution_tables 合并续算；summarize_transaction_pnl 使用截止日保存的
lot/mark snapshot 和成熟事件。不能把最新 open_lots/transaction_pnl 直接用于
旧窗口，不能读未来价格/成交或回放账户。每个 session/order/planning 支持取消。

每个计划的 target/actual return 等于该计划归因 P&L 除以其已成交金额；金额收益和费用列单独展示。账户收益率按权益/NAV 计算。
