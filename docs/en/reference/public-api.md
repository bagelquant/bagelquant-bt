# Public API

The stable public API is exported from `bagelquant_bt`. Version 0.3 accepts
strongly typed predictions only; ordinary `Panel`, raw DataFrames, and direct
weights are not public backtest inputs.

## Entry points

```python
from bagelquant_bt import compose_prediction, run_daily_rank_path_diagnostics, run_prediction_backtest, run_prediction_evaluation, run_prediction_horizon_diagnostics
```

- `compose_prediction(...) -> PredictionPanel` applies a `PredictionComposer` at the
  cadence selected by `AlphaPolicy`. Supervised composers derive labels
  from one execution price to the next and enforce label-availability cutoffs.
- `run_prediction_backtest(prediction, prices, calendar, weight_policy=..., ...)`
  schedules a `PredictionPanel`, applies `ExecutionPolicy` and `WeightPolicy`,
  and returns `BacktestResult`.
- `run_prediction_evaluation(scheduled_signal, prices, ...)` computes IC,
  quantiles, lag diagnostics, and signal-driven portfolio results.
- `run_prediction_horizon_diagnostics(scheduled_signal, prices, ...)` computes
  one fixed-session label window at a time and retains aggregate IC,
  centered-rank Book, gross-one Tail,
  quantile structure, signal persistence, HAC inference, BH q-values, and
  staggered cohorts without constructing portfolio performance. Structurally
  valid Book/Tail weights remain fixed when a member label is missing: that
  contribution is zero while coverage counts continue to expose the gap.
- `run_daily_rank_path_diagnostics(scheduled_signal, prices, config, ...) -> DailyRankPathDiagnostics`
  calculates capital-free daily Book/Tail gross/net diagnostic paths, where Net
  subtracts requested-turnover proportional costs but excludes capital, minimum
  fees, execution blocks, and insolvency; it also reports Book requested/executed
  turnover with an initial-rebalance marker, common-sample Book lead-lag returns
  for integer lags `-30..30`, and Book/Tail lag paths at
  `0/1/2/5/10/20/60`.
- `rolling_window_information_coefficients` and `implied_signal_half_life`
  expose the causal 240-valid-observation rolling IC and per-lag half-life
  primitives used by daily result charts.
- `session_window_forward_returns`, `centered_rank_book_weights`,
  `gross_one_tail_weights`, `hac_mean_test`, and
  `non_overlapping_cohort_statistics` expose the corresponding deterministic
  primitives.
- `quantile_rank_information_coefficients(quantile_returns, *, periods=None)`
  derives the monotonic rank IC from stored q1-to-qN gross returns. Optional
  `time`/`next_time` periods compound daily returns into one observation per
  complete execution interval.
- `summary_report(...)` builds a static HTML report for a backtest or signal
  evaluation result.

Candidate-prediction validation is available through `score_ic_validation`,
`select_top_n_stable`, `top_n_monthly_performance`, and
`score_top_n_performance`. Undefined or constant-prediction IC months are
excluded rather than scored as zero; candidates fail below their minimum valid
month count. Top-N selection always returns exactly N eligible assets using
prediction descending then asset ID ascending for cutoff ties, and reports the
cutoff/tie audit. Turnover regularization is explicitly
`net_sharpe - lambda * average_turnover`.
`top_n_monthly_performance` accepts either a compact proportional-turnover
cost or detailed commission, per-asset minimum fee, sell-tax, slippage, and
initial-capital inputs.

`AlphaPolicy` and `ExecutionPolicy` are separate contracts. A weight policy
receives `PredictionPanel` and returns
`WeightBuild(weights: Panel, skipped: DataFrame)`. The standalone
`allocate_integer_positions` helper converts one continuous target snapshot to
whole-lot positions with explicit prices, budgets, lot sizes, and frozen
minimums; market-specific rules and live
order submission are outside the package boundary.

## Configuration

```python
from bagelquant_bt import BacktestConfig, TransactionCostConfig

config = BacktestConfig(
    initial_capital=1_000_000,
    transaction_cost=TransactionCostConfig(
        rate=0.00015,
        min_fee=5.0,
        buy_slippage_rate=0.0005,
        sell_slippage_rate=0.0005,
        stamp_tax_rate=0.0005,
    ),
    annualization=252,
    quantiles=5,
    top_n=50,
)
```

`initial_capital` must be positive. `quantiles` and `top_n` control evaluation
metrics; portfolio construction parameters belong to the selected
`WeightPolicy`.
`insolvency_action` defaults to `"raise"`. Setting it to `"freeze_zero"`
caps effective fees at available wealth on the insolvency session, records
requested and unfunded fees, sets net return to `-100%`, and freezes later
gross/net returns and trading at zero. Return, lag, and quantile paths expose
`is_bankrupt` and `bankruptcy_event` markers.

## Results

`BacktestResult` exposes weights, returns, value, turnover, transaction costs,
performance, execution blocks, coverage, and missing price keys.

`PredictionEvaluationResult` exposes the evaluated prediction, execution-to-execution
forward returns, Pearson and Spearman IC, quantile and spread results, TOP N
portfolios, lag analysis, IC decay, benchmarks, coverage, and missing price
keys. `FactorEvaluationResult` remains the internal result class name for the
statistical implementation; operator-facing APIs use Prediction terminology.

## Exceptions

- `BagelQuantBacktestError`: base package error.
- `BacktestConfigError`: invalid configuration.
- `InputValidationError`: invalid or incompatible market data.
