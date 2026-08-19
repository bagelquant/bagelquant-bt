import math
from datetime import date

import polars as pl

from bagelquant_bt.performance import summarize_performance
from bagelquant_bt.results import TransactionCostBreakdown
from bagelquant_bt.window import _single_return_metrics


def test_negative_terminal_wealth_has_no_real_annualized_return() -> None:
    returns = pl.DataFrame(
        {
            "time": [date(2024, 1, 2), date(2024, 1, 3)],
            "gross_return": [-1.5, 0.0],
            "net_return": [-1.0, 0.0],
        }
    )

    summary, matrix = summarize_performance(
        returns=returns,
        turnover=pl.DataFrame(),
        costs=TransactionCostBreakdown(pl.DataFrame()),
        initial_capital=100.0,
        annualization=2,
    )

    assert summary.final_gross_value == -50.0
    assert summary.gross_total_return == -1.5
    assert math.isnan(summary.gross_annualized_return)
    assert summary.final_net_value == 0.0
    assert summary.net_annualized_return == -1.0
    assert math.isnan(
        matrix.filter(pl.col("metric") == "annualized_return").item(0, "gross")
    )
    assert _single_return_metrics([-1.5, 0.0], 2)["annualized_return"] is None
