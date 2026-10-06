# 成本

权重研究 cost_return = 0.0005 × Net 调仓前漂移权重与 target 的 full-L1 差，
包括首次建仓，直接从本期收益扣除。虚拟 research cash 不要求资金/lot/min-fee。
Gross/Net 分别漂移；net_pre_cost_return − cost_return = net_return，不能要求
独立 Gross path 减当期成本恒等于 Net path。

执行默认每订单/交易日 max(成交金额 × 0.0005, 5)，无成交不收费，独立 plan ID
分别计费；跨日 residual retry 再收费。默认不含税/slippage。调用方可显式设置
买卖滑点、卖出税、transfer fee、lot/settlement。可买股数预算包括全部钱款。

自定义纯规则须稳定 ID/version/parameters，quote 返回有效价格/非负费用，
买入总支出随股数单调增加。理论 P&L、实际成交价 P&L 和显式费用分别报告；
implementation shortfall 也包含未成交与价格差。FIFO 按匹配股数分摊成本。

原生 stress 只放大声明分量，比例佣金不改变 min-fee/tax，零 slippage 仍为零。
不透明的 custom rule 需调用方明确提供场景，不能推测其费用构成。
