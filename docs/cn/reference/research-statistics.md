# 研究检验与比较

这些公开原语消费明确选择的已保存数据表，不拟合模型、不模拟账户、不选择生产候选，
也不管理验证批次。协议、试验范围、来源凭据与不可变结果引用由 Workbench 冻结。

`library_endpoint_tests(trials, q_max=0.10)` 接收各主终点的双侧 HAC p 值，对全部有效
检验结果执行 BH，包括后来放弃的试验，同时检查预先声明的方向和最低效应。
缺少统计量时保留空 p/q 值及不可用原因。p 值为 0.01、0.04 时，BH q 值为 0.02、0.04；
第三个失败且没有 p 值的试验单独列出，不增加有效检验分母。

`deflated_sharpe(returns, target=..., trials=..., minimum_sessions=240)` 接收长表
`(time, trial, return)` 成本后收益，使用共同有限样本，保留声明试验范围与相关矩阵，
按 `rho + (1-rho)*M` 估计有效试验数。至少三个试验，默认至少 240 个 session，且样本数
必须超过试验数。退化收益、不支持的负平均相关、奇异或病态的相关矩阵，以及有效试验数
不足两个时，结果明确不可用。这一辅助估计不代表未登记的历史试验。公式及相关近似见
[Bailey 和 López de Prado（2014）](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)。

`common_sample_comparison` 在共同有限日期上比较逐日原语。`common_prediction_ic`
在共同有限资产／日期坐标上计算增量及条件秩 IC；完全共线的变体没有独立信号。
`account_fill_turnover` 使用实际成交金额和前一日权益，首日使用冻结的初始资金。

`capacity_participation` 将逐笔绝对人民币成交金额除以前 20 个市场 session 的资产平均
人民币成交额。每个 session 都须有非负有限观测，已知零成交额计入均值；排除执行日金额。
覆盖不足或平均成交额为零时参与率为空。
账户压力场景使用现有保存目标账户引擎，各自保存现金、持仓、成交与检查点。
资金因素同时缩放初始资金和固定名义资金目标，避免固定资金账户立即撤回新增资金。
