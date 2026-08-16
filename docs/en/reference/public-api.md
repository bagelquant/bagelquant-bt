# Public API

The stable public API is exported from `bagelquant_bt`. Version 0.3 accepts
strongly typed predictions only; ordinary `Panel`, raw DataFrames, and direct
weights are not public backtest inputs.

## Entry points

```python
from bagelquant_bt import compose_prediction, run_prediction_backtest, run_prediction_evaluation
```

- `compose_prediction(...) -> PredictionPanel` applies a `PredictionComposer` at the
  cadence selected by `AlphaPolicy`. Supervised composers derive labels
  from one execution price to the next and enforce label-availability cutoffs.
- `run_prediction_backtest(prediction, prices, calendar, weight_policy=..., ...)`
  schedules a `PredictionPanel`, applies `ExecutionPolicy` and `WeightPolicy`,
  and returns `BacktestResult`.
- `run_prediction_evaluation(scheduled_signal, prices, ...)` computes IC,
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

`AlphaPolicy` and `ExecutionPolicy` are separate contracts. A weight policy
receives `PredictionPanel` and returns
`WeightBuild(weights: Panel, skipped: DataFrame)`. Quantity sizing and live
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
