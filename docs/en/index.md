# bagelquant-bt documentation

BT evaluates saved Alpha/Prediction values, portfolio weights and frozen share
transaction plans. It supplies financial mechanics and reusable statistics;
Core owns value/model computation and Data owns data and PIT evidence.

- [Quick start](quick-start.md)
- [Architecture and artifact ownership](architecture.md)
- [Workers and resource limits](performance.md)
- [Public API](reference/public-api.md)
- [Alpha evaluation](reference/factor-evaluation.md)
- [Execution and FIFO](reference/account-backtest.md)
- [Costs](reference/transaction-costs.md)
- [Research inference](reference/research-statistics.md)

The three public modes share one implementation per concern. Removed runners,
policy registries and legacy adapters have no compatibility aliases. Real
cutover is a separately authorized later stage.
