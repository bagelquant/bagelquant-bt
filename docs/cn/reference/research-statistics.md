# 保存结果的通用统计

return_statistics 统一计算收益、波动、Sharpe、回撤、Calmar、hit rate；
风险拟合/profile/linking、factor/exposure 等函数只消费明确的保存输入，不训练
模型、不访问 provider、不改变治理。

HAC/BH 使用双侧 p-value，并纳入全部有效声明 trials（含放弃试验）；
无效样本保留 unavailable reason。Deflated Sharpe 需要明确 trial scope 和共同
finite 样本，其 correlation/effective-trials 限制可见。Workbench 选择协议/治理。

common_sample_comparison、partial_rank_ic、incremental_ic_summary、
factor_return_correlation、holdings_factor_exposure 复用保存坐标/收益/持仓。
account_fill_turnover 使用真实 notional 和 prior equity。default/custom period
只聚合保存 primitives 并重新核实成熟度，不回放账户或模型。

capacity_participation 取执行日前 20 个 session 的成交金额，不含执行日。
已知非负零值有效；缺失/负/nonfinite 或均值为零不可用。这是 capacity 诊断，
不是成交限制。execution_stress_scenarios 提供 9 个原生单因子场景，账本独立。
