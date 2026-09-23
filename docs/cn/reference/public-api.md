# 公开 API

稳定 API 从 `bagelquant_bt` 导出。0.2 只接受强类型 Signal；普通 `Panel`、
裸 DataFrame 和直接 weights 均不能进入公开回测入口。

## 入口函数

```python
from bagelquant_bt import compose_prediction, run_daily_rank_path_diagnostics, run_prediction_backtest, run_prediction_evaluation, run_prediction_horizon_diagnostics
```

- `compose_prediction(...) -> PredictionPanel`：按 `AlphaPolicy` 的 cadence
  执行 `PredictionComposer`。监督式 composer 使用本次 execution 到下一次
  execution 的收益，并检查标签可用时间。
- `run_prediction_backtest(...)`：依次应用 `AlphaPolicy`、
  `ExecutionPolicy`、`WeightPolicy` 和内部 weights 引擎。
- `run_prediction_evaluation(scheduled_signal, prices, ...)`：计算 IC、分位数、
  lag、IC decay 和 Signal 驱动的组合结果。
- `run_prediction_horizon_diagnostics(scheduled_signal, prices, ...)`：在不构造组合绩效的
  前提下计算固定 session 前向收益、IC、centered-rank Book、gross-one Tail、quantile
  结构、信号持久性、HAC 推断、BH q-value 与 staggered cohorts。Book/Tail 权重结构有效后，
  成员标签缺失时保留原权重并把该成员贡献记为 0，同时 coverage 计数继续公开缺口。
- `run_daily_rank_path_diagnostics(scheduled_signal, prices, config, ...) -> DailyRankPathDiagnostics`：
  模拟每日 Book/Tail gross/net 诊断路径、Book requested/executed turnover，以及
  初始建仓标记、`-30..30` 整数 lag 的共同样本 Book lead-lag return，并输出
  `0/1/2/5/10/20/60` 的 Book/Tail lag 路径。
- `rolling_window_information_coefficients` 与 `implied_signal_half_life` 公开日频图表使用的
  因果 240-valid-observation rolling IC 与逐 lag half-life primitive。
- `session_window_forward_returns`、`centered_rank_book_weights`、
  `gross_one_tail_weights`、`hac_mean_test` 与
  `non_overlapping_cohort_statistics` 公开对应的确定性 primitive。
- `quantile_rank_information_coefficients(quantile_returns, *, periods=None)`：从 q1 到
  qN 的 gross 组收益生成单调性 rank IC；可选的 `time`/`next_time` periods 会先把逐日
  收益压缩为每个完整 execution 区间一个观测。

候选预测验证由 `score_ic_validation`、`select_top_n_stable`、
`top_n_monthly_performance` 和 `score_top_n_performance` 提供。IC 未定义或 prediction
为常数的月份不会按零计分；有效月份不足时候选无效。Top-N 对 cutoff tie 使用 prediction
降序、asset ID 升序，稳定选取恰好 N 只并返回 cutoff/tie 审计。换手正则目标明确定义为
`net_sharpe - lambda * average_turnover`。
`top_n_monthly_performance` 可以使用紧凑的比例换手成本，也可以分别接收佣金、逐资产
最低佣金、卖出税、滑点与初始资金。

`WeightPolicy` 接收 `ScheduledPrediction`，返回
`WeightBuild(weights: Panel, skipped: DataFrame)`。独立的
`allocate_integer_positions` 接口以显式价格、预算、整手大小和冻结最低数量，把一期连续目标转换为
整数手数仓位；大截面会先预分配连续目标附近的基准仓位，再在最后四手的有界范围内按跟踪误差
顺序用确定性堆补齐，避免资金部署问题退化成耗时不可控的子集和 MILP。小截面仍保留精确的两阶段
MILP。市场专属规则与实盘报单不属于本包边界。

`BacktestConfig.insolvency_action` 默认为 `"raise"`，保持严格失败语义。设为
`"freeze_zero"` 后，资不抵债当日的有效费用封顶为可用财富，同时记录请求费用和未支付
费用，净收益记为 `-100%`；之后 gross/net 收益与交易均冻结为零。return、lag 和
quantile 路径会提供 `is_bankrupt` 与 `bankruptcy_event` 标记。

`BacktestResult` 包含权重、收益、净值、换手、成本、执行阻塞和覆盖度。
`PredictionEvaluationResult` 包含 Signal、execution-to-execution forward returns、
Pearson/Spearman IC、分位数、spread、TOP N、lag、IC decay 与基准结果。

## 暴露约束优化

`PredictionExposureConstrainedOptimizerPolicy(concentration_penalty,
turnover_penalty, max_weight, exposure_bounds={}, max_turnover=None)` 是独立的
仅多头、全投资约束优化器。每个暴露列对应一个 `ExposureBounds(lower=None, upper=None)`，
至少提供一个有限边界。调用 `build(prediction, reference_weights=..., exposures=...)`，
传入以 `(time, asset_id)` 为键的 PIT 个股暴露；行业可使用调用方明确构造的 0/1 列。
BT 不推断具体因子名称，也不读取市场数据。

目标函数为预测收益减集中度平方惩罚与 L1 换手惩罚。换手使用完整权重变动绝对值之和，
包含有限 Prediction 截面之外参考持仓的强制退出，不除以二；初始全投资需要换手预算一。
约束针对目标权重，不代表整手和成交阻塞后的实际仓位始终满足约束。

安装 `bagelquant-bt[optimizer]`；CVXPY/CLARABEL 仅求解时加载。暴露缺失/非有限、约束
不可行或求解失败/不精确均携日期报错，不放宽边界、不退回等权。
`WeightBuild.diagnostics` 返回预测尺度、目标函数分项、换手、强制退出、求解状态以及
逐项暴露和上下界余量。原解析优化器数值行为保持不变。

Portfolio Path identity v4 使用必填 `PortfolioPathIdentity.pipeline`，删除原 Combo 字段。
调用方传入冻结链路身份；旧 v3 路径缓存不得自动采用。

## 异常

- `BagelQuantBacktestError`：包级基础异常。
- `BacktestConfigError`：配置无效。
- `InputValidationError`：市场数据无效或不兼容。
