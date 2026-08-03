# Public API

The stable public API is exported from `bagelquant_bt`. Version 0.2 accepts
strongly typed signals only; ordinary `Panel`, raw DataFrames, and direct
weights are not public backtest inputs.

## Entry points

```python
from bagelquant_bt import compose_signal, run_signal_backtest, run_signal_evaluation
```

- `compose_signal(...) -> SignalPanel` applies a `SignalComposer` at the
  cadence selected by `SignalDatePolicy`. Supervised composers derive labels
  from one execution price to the next and enforce label-availability cutoffs.
- `run_signal_backtest(signal, prices, calendar, signal_date_policy, ...)`
  schedules a `SignalPanel`, applies `ExecutionPolicy` and `PortfolioPolicy`,
  and returns `BacktestResult`.
- `run_signal_evaluation(scheduled_signal, prices, ...)` computes IC,
  quantiles, lag diagnostics, and signal-driven portfolio results.
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

`SignalDatePolicy` and `ExecutionPolicy` are separate contracts. A portfolio
policy receives `ScheduledSignal` and returns
`PortfolioBuild(weights: Panel, skipped: DataFrame)`. `OrderSizingPolicy` and
`OrderPlan` reserve the later target-weight-to-order boundary; quantity sizing
is not implemented in 0.2.

## Configuration

```python
from bagelquant_bt import BacktestConfig, TransactionCostConfig

config = BacktestConfig(
    initial_capital=1_000_000,
    transaction_cost=TransactionCostConfig(
        rate=0.00015,
        min_fee=5.0,
        slippage_rate=0.0005,
        stamp_tax_rate=0.0005,
    ),
    annualization=252,
    quantiles=5,
    top_n=50,
)
```

`initial_capital` must be positive. `quantiles` controls signal buckets and
`top_n` controls the default equal-weight portfolio policy.

## Results

`BacktestResult` exposes weights, returns, value, turnover, transaction costs,
performance, execution blocks, coverage, and missing price keys.

`SignalEvaluationResult` exposes the evaluated signal, execution-to-execution
forward returns, Pearson and Spearman IC, quantile and spread results, TOP N
portfolios, lag analysis, IC decay, benchmarks, coverage, and missing price
keys. `FactorEvaluationResult` remains the internal result class name for the
statistical implementation; operator-facing APIs use Signal terminology.

## Exceptions

- `BagelQuantBacktestError`: base package error.
- `BacktestConfigError`: invalid configuration.
- `InputValidationError`: invalid or incompatible market data.
