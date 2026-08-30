# Signal 评估

Signal 评估会把 `ScheduledPrediction` 中的每个 `PredictionPanel` 快照解释为截面
prediction，数值越高代表越正向。

## 固定预测期限

`run_prediction_horizon_diagnostics` 是只评估预测能力的入口，将 prediction 更新频率与
经济预测期限分开。固定日频协议从映射后的 next-open execution price 开始，计算累计
`1/5/10/20/40/60/120D`，以及互不重叠的 `1D`、`2–5D`、`6–20D`、`21–60D`、
`61–120D` bucket。尚未完成的尾部标签不进入统计，并通过逐 window coverage 公开；
系统不会按未来价格可得性筛选信号截面或重新归一化权重。

主要连续截面收益先计算平均并列 percentile rank，在评价日 Universe 内中心化，再按中心化
分数绝对值之和归一化。所得 Book 的净敞口为 0、gross 为 1，多空两侧分别为
`+0.5/-0.5`。常数截面或无法形成双侧时，`centered_rank_book_weights` 返回 unavailable。
`gross_one_tail_weights` 在相同 gross-one 口径下做多 q1、做空 q10。Book、Tail 和完整
quantile curve 都是预测诊断，不生成 NAV、成本、换手、Sharpe 或回撤。

Book 或 Tail 的权重结构有效后，某成员 forward return 缺失时，该成员当期贡献记为 0，
但原始权重仍保留；该行结果继续可用，`expected_count`、`observed_count` 与
`coverage_ratio` 仍如实公开标签缺口。系统不会剔除成员或重新计算权重。这个规则不放宽
十个 quantile 的完整共同样本要求，也不改变 IC 与其他诊断的输入口径。

每个 window 输出 Pearson/Spearman IC、正 IC 比例、ICIR、Book、Tail、quantile-rank IC
和截面回归斜率。均值推断统一使用 Bartlett Newey–West，lag 为
`max(window_width-1, floor(4*(n/100)^(2/9)))`，并输出双侧 p-value 与 95% CI。
同一指标族的全部十二个 window 统一进行 Benjamini–Hochberg 校正；全部按 window width
取模的 staggered non-overlapping cohorts 保留为稳健性检查。Signal rank persistence 在
`1/5/10/20/40/60/120D` 计算；half-life 只报告首次跨过 0.5 的网格区间，或 `>120D`。

下文旧 `run_prediction_evaluation` 属于组合评估 API，与固定期限预测诊断相互独立。

## 日频诊断排名路径

`run_daily_rank_path_diagnostics` 把每日 PIT 信号转换为同一 centered-rank Book 与
gross-one Tail 目标并直接计算去账户化研究收益。输出包含 Book/Tail 每日 gross/net return、
以及十个持续的日度再平衡等权分位 Gross 路径。每组覆盖约十分之一有效信号截面并在映射的
next-open 生效；单只股票缺失收益时按冻结为零、恢复日补记处理，不会让整组失效，也不会重选或
重新归一化其余股票。
Book requested/executed turnover，以及 `-30` 到 `+30` 每个整数 lead/lag 的 Book
gross/net return。初始建仓计入换手与成本，并由 `is_initial_rebalance` 精确标记，因此读取
任意日期窗口时只排除这一次真实初始建仓，不会误删截取窗口的首行。
`alpha_return_lag_returns` 还输出 Book/Tail 在 `0/1/2/5/10/20/60` lag 下的 gross/net
路径，十四条路径共用同一完整日期样本。负 lag 会在信号出现前交易，必须明确标为
非 PIT/look-ahead 诊断；所有 lag 都只使用平移后 execution date 的共同完整重叠区间。

日频 Summary 的 autocorrelation 使用 `1..120` 全部 lag 的平均并列横截面 rank
correlation。逐 lag implied half-life 仅在 `0 < rho_lag < 1` 时按
`-lag * ln(2) / ln(rho_lag)` 计算。Rolling IC 严格使用 240 个有效 observation，保持因果，
warm-up 未完成时不输出数值。这些结果只服务诊断图表，不会把 Alpha 结果变成账户或真实
Weight Policy 的 Portfolio Performance section。

应用同时需要两类结果时使用 `run_daily_prediction_diagnostics`。它只校验并 collect 一次
Signal，只准备一次市场、交易日历、Book、Tail 与 quantile 权重，逐 window 聚合标签，并且
只计算一次 `1..120` rank persistence，再把所需子集复用于期限结果。返回值为
`DailyPredictionDiagnostics(horizons, paths)`；两个独立入口继续保留相同 schema，供只请求
单类诊断的场景使用。

## IC 和 ICIR

对每个信号期，`bagelquant-bt` 计算 Signal 与本次 execution price 到下一次
Signal execution price 收益之间的截面相关性。

Signal 评估会在 `result.ic` 中同时输出 Pearson 相关和 Spearman 秩相关：

```python
result.ic.select("time", "pearson_ic", "spearman_ic")
```

`result.ic_summary` 包含每种方法的均值、标准差和 ICIR。兼容字段
`ic_mean`、`ic_std` 和 `icir` 使用 Spearman IC。

`icir` 定义为：

```text
mean(IC) / standard_deviation(IC) * sqrt(每年 IC 观测数)
```

`BacktestConfig.ic_annualization` 用于指定每年的 IC 观测数；未指定时，默认使用日收益的 `annualization`。这与组合收益指标不同：年化收益率、波动率、Sharpe、滚动指标、基准和滞后 Sharpe 都使用日收益年化数。

## Signal 与 Execution Policy

`AlphaPolicy.select` 从 `PredictionPanel` 选择完整截面，返回带调度、执行日期
lineage 和强类型 signal 的 `ScheduledPrediction`。`month_end` 优先选择月末最后一个
开市交易日；若整张截面不存在，只能回退到同一自然月内最近的此前完整截面，
否则记录 skip。单个资产绝不会从此前日期回填。

`ExecutionPolicy("next_open")` 独立把再平衡日映射到下一开市交易日。`run_prediction_evaluation` 接收已经解析的统一 selection，IC 使用本次 execution date 到下一次 execution date 的收益，组合则在两次执行之间逐日估值；最后一条没有完整后续执行期的信号不进入 IC。

因此月频 `month_end` Policy 使用自然月末选出的值，对应其映射的 next-open execution price
至下一次 scheduled execution price 的完整收益；尚未结束的当前月份不会进入 IC 或监督式
Composer 的滚动窗口。

## 分位数组合收益

每天按因子分数从高到低排序资产，并切分为若干分位数组：`q1` 为最高分数组，`qN` 为最低分数组。每个分位数组合收益是组内资产前向收益的等权平均。

spread 为：

```text
q1 收益 - qN 收益
```

日频 prediction 的 Quantile Test 使用
`DailyRankPathDiagnostics.quantile_returns` 中独立的持续目标组合序列，不再把固定期限标签均值
拼接成组合路径。十组使用相同的结构有效日期；有效资产少于十只、常量信号或不存在下一交易日时，
十组在该日共同 unavailable。成本、执行限制、现金与账户状态仍全部排除。

`quantile_rank_information_coefficients(quantile_returns, periods=...)` 从制品中的 q1 到
qN 标签推断 N，因此历史 q5 结果仍可读取。提供 execution periods 后，函数先把每组在
`[time, next_time)` 内的逐日 gross return 复合为一个区间收益，并在区间起点生成唯一观测；
再将 q1 到 qN 赋分 N 到 1，与组收益计算 Spearman 相关。组收益严格递减为 `+1`，严格
递增为 `-1`。缺组、任一组没有有限收益或组收益完全相同时返回 null。结果统计检验使用
与普通 IC 相同的完整区间和双侧单样本 Student t-test。

结果表、图表、图例和 HTML 报告统一按标签数字部分排序为 `q1, q2, ..., q10`；历史 q5
制品仍保持 `q1, ..., q5`。

## TOP N 回测

TOP N 回测会把因子分数转换成长-only 等权组合：

```text
每天前 N 个资产 -> 每个资产 1 / N 权重
```

生成的权重表会进入与普通权重回测相同的引擎，包括交易成本计算。

## Spread 和滞后分析

Signal 评估还会构造 spread 组合：做多 `q1`、做空 `qN`，并通过同一个含交易成本的回测引擎计算结果。

`lag_analysis` 会评估 TOP N 和 spread 组合在 Signal 滞后 0、1、2、3、4、5、10、20、30、60 个交易日后的累计收益和 Sharpe；月频 Signal 的 Lag 15 是 15 个开市日，不是 15 个按月观测。

`lag_returns` 包含相同组合和滞后下的 gross/net 累计收益时间序列。

`ic_decay` 会在相同交易日滞后上报告 Pearson 和 Spearman 的平均 IC，并以 IC decay 折线图展示。
