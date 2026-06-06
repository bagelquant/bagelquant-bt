from __future__ import annotations

import pandas as pd
import pytest

from bagelquant_bt import BacktestConfig, run_backtest, run_weight_backtest
from bagelquant_bt.exceptions import BacktestConfigError, InputValidationError


def prices() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [100.0, 110.0, 121.0], "b": [100.0, 90.0, 99.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


def weights() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [1.0, 0.0, 1.0], "b": [0.0, 1.0, 0.0]},
        index=prices().index,
    )


def test_run_backtest_dispatches_weights() -> None:
    result = run_backtest(
        weights(),
        prices(),
        kind="weights",
        config=BacktestConfig(initial_capital=100_000),
    )

    assert result.gross_returns.round(6).tolist() == [0.1, 0.1]
    assert hasattr(result, "net_returns")


def test_weight_backtest_provides_gross_and_net_results() -> None:
    result = run_weight_backtest(
        weights(),
        prices(),
        config=BacktestConfig(initial_capital=100_000),
    )

    assert result.gross_cumulative_returns.iloc[-1] > 0
    assert (
        result.net_cumulative_returns.iloc[-1]
        < result.gross_cumulative_returns.iloc[-1]
    )
    assert result.summary.final_gross_value > result.summary.final_net_value


def test_config_is_required_for_minimum_fee_mode() -> None:
    with pytest.raises(BacktestConfigError, match="config is required"):
        run_weight_backtest(weights(), prices())


def test_invalid_kind_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="kind"):
        run_backtest(
            weights(),
            prices(),
            kind="portfolio",
            config=BacktestConfig(initial_capital=100_000),
        )
