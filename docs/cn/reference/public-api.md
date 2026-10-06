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
