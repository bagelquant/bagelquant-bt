from __future__ import annotations

import pandas as pd

from bagelquant_bt import BacktestConfig, run_backtest, run_factor_evaluation
from bagelquant_bt.factor import top_n_equal_weights


def factor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [3.0, 1.0, 3.0],
            "b": [2.0, 2.0, 2.0],
            "c": [1.0, 3.0, 1.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


def prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [100.0, 110.0, 110.0],
            "b": [100.0, 100.0, 100.0],
            "c": [100.0, 90.0, 99.0],
        },
        index=factor().index,
    )


def test_factor_evaluation_ic_icir_and_quantiles() -> None:
    result = run_factor_evaluation(
        factor(),
        prices(),
        config=BacktestConfig(initial_capital=100_000, quantiles=3, top_n=1),
    )

    assert result.ic.iloc[0] > 0.9
    assert result.ic_mean == result.ic.mean()
    assert set(result.quantile_returns.columns) == {"q1", "q2", "q3"}
    assert "q3" in result.quantile_cumulative_returns
    assert result.top_minus_bottom.iloc[0] > 0


def test_top_n_equal_weight_portfolio_generation() -> None:
    weights = top_n_equal_weights(factor(), top_n=2)

    assert weights.iloc[0].tolist() == [0.5, 0.5, 0.0]
    assert weights.iloc[1].tolist() == [0.0, 0.5, 0.5]


def test_run_backtest_dispatches_factor() -> None:
    result = run_backtest(
        factor(),
        prices(),
        kind="factor",
        config=BacktestConfig(initial_capital=100_000, quantiles=3, top_n=1),
    )

    assert (
        result.top_n_backtest.net_returns.iloc[0]
        < result.top_n_backtest.gross_returns.iloc[0]
    )
