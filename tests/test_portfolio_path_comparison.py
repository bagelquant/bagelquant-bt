from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from bagelquant_bt import (
    AccountBacktestConfig,
    TransactionCostConfig,
    compare_portfolio_paths,
    run_account_backtest,
    run_actual_performance_path,
    run_continuous_target_path,
    run_total_return_weight_paths,
    summarize_portfolio_path_returns,
)


def _cost(rate: float = 0.0) -> TransactionCostConfig:
    return TransactionCostConfig(
        rate=rate,
        min_fee=0.0,
        buy_slippage_rate=0.0,
        sell_slippage_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
    )


def _account(*, cost: TransactionCostConfig):
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    targets = pl.DataFrame(
        {"time": [days[0]], "asset_id": ["a"], "weight": [1.0]}
    )
    prices = pl.DataFrame(
        {
            "time": days,
            "asset_id": ["a"] * 3,
            "open": [10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0],
        }
    )
    account = run_account_backtest(
        targets,
        prices,
        corporate_action_coverage=pl.DataFrame(
            {"time": days, "is_complete": [True] * 3}
        ),
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=1_000.0,
            transaction_cost=cost,
        ),
    )
    total_return = prices.select(
        "time", "asset_id", pl.col("open").alias("total_return_price")
    )
    target = run_continuous_target_path(
        targets,
        total_return,
        initial_capital=1_000.0,
        transaction_cost=cost,
    )
    actual = run_actual_performance_path(
        account.fills,
        account.positions,
        total_return,
        initial_capital=1_000.0,
    )
    return target, actual, account


def test_zero_cost_divisible_paths_are_identical() -> None:
    target, actual, account = _account(cost=_cost())

    comparison = compare_portfolio_paths(
        target, actual, account, initial_capital=1_000.0
    )

    nav = comparison.nav_paths
    assert nav.get_column("target_gross_nav").to_list() == pytest.approx(
        nav.get_column("target_net_nav").to_list()
    )
    assert nav.get_column("target_gross_nav").to_list() == pytest.approx(
        nav.get_column("actual_gross_nav").to_list()
    )
    assert nav.get_column("actual_gross_nav").to_list() == pytest.approx(
        nav.get_column("actual_net_nav").to_list()
    )
    assert comparison.summary.equals(
        summarize_portfolio_path_returns(nav, annualization=252)
    )


def test_portfolio_path_summary_recomputes_the_selected_return_window() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    returns = pl.DataFrame(
        {
            "time": days,
            "target_gross_return": [0.10, -0.10, 0.20],
            "target_net_return": [0.05, -0.05, 0.10],
            "actual_gross_return": [0.08, -0.08, 0.16],
            "actual_net_return": [0.04, -0.04, 0.08],
        }
    )

    full = summarize_portfolio_path_returns(returns, annualization=2)
    selected = summarize_portfolio_path_returns(
        returns.filter(pl.col("time").is_between(days[1], days[2])),
        annualization=2,
    )
    full_target = full.filter(pl.col("path") == "target_gross").row(0, named=True)
    selected_target = selected.filter(pl.col("path") == "target_gross").row(
        0, named=True
    )

    assert full_target["cumulative_return"] == pytest.approx(0.188)
    assert selected_target["cumulative_return"] == pytest.approx(0.08)
    assert selected_target["annualized_return"] == pytest.approx(0.08)
    assert selected_target["annualized_volatility"] == pytest.approx(0.3)
    assert selected_target["sharpe"] == pytest.approx(1 / 3)


def test_actual_net_uses_only_explicit_fill_costs() -> None:
    target, actual, account = _account(cost=_cost(rate=0.01))

    comparison = compare_portfolio_paths(
        target, actual, account, initial_capital=1_000.0
    )

    assert actual.returns.get_column("explicit_cost").sum() == pytest.approx(
        account.fills.select(
            pl.sum_horizontal(
                "commission", "stamp_tax", "transfer_fee", "slippage_cost"
            ).sum()
        ).item()
    )
    assert comparison.nav_paths.get_column("actual_net_nav")[-1] < (
        comparison.nav_paths.get_column("actual_gross_nav")[-1]
    )
    assert comparison.reconciliation.columns == [
        "time",
        "performance_nav",
        "execution_equity",
        "execution_equity_index",
        "total_gap",
        "dividend_cash_effect",
        "share_action_effect",
        "corporate_action_timing_effect",
        "lot_rounding_effect",
        "residual_cash_effect",
        "settlement_timing_effect",
        "fee_accounting_effect",
        "position_valuation_effect",
        "unexplained_gap",
    ]


def test_total_return_target_holds_units_between_targets() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    path = run_continuous_target_path(
        pl.DataFrame(
            {
                "time": [days[0], days[2]],
                "asset_id": ["a", "a"],
                "weight": [1.0, 0.5],
            }
        ),
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a"] * 3,
                "total_return_price": [10.0, 11.0, 12.0],
            }
        ),
        initial_capital=1_000.0,
        transaction_cost=_cost(),
    )

    assert path.returns.get_column("gross_return").to_list() == pytest.approx(
        [0.0, 0.1, 1 / 11]
    )


def test_missing_execution_mark_defers_entry_without_using_future_price() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    path = run_continuous_target_path(
        pl.DataFrame(
            {"time": [days[0]], "asset_id": ["a"], "weight": [1.0]}
        ),
        pl.DataFrame(
            {
                "time": [days[0], days[1], days[1], days[2], days[2]],
                "asset_id": [
                    "calendar-anchor",
                    "a",
                    "calendar-anchor",
                    "a",
                    "calendar-anchor",
                ],
                "total_return_price": [1.0, 10.0, 1.0, 11.0, 1.0],
            }
        ),
        initial_capital=1_000.0,
        transaction_cost=_cost(),
    )

    # The missing execution-day mark remains cash.  Entry happens only when
    # the price is observed on day two; no future mark is pulled backward.
    assert path.returns.get_column("gross_return").to_list() == pytest.approx(
        [0.0, 0.0, 0.1]
    )


def test_quantile_style_weights_are_not_daily_rebalanced() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    result = run_total_return_weight_paths(
        {
            "q1": pl.DataFrame(
                {
                    "time": [days[0], days[0]],
                    "asset_id": ["a", "b"],
                    "weight": [0.5, 0.5],
                }
            )
        },
        pl.DataFrame(
            {
                "time": days * 2,
                "asset_id": ["a"] * 3 + ["b"] * 3,
                "total_return_price": [10.0, 20.0, 20.0, 10.0, 10.0, 10.0],
            }
        ).sort("time", "asset_id"),
    )

    # Day two grows from 1 to 1.5.  With unchanged units day three is flat;
    # resetting to equal weights every day would alter the exposure.
    assert result.get_column("gross_return").to_list() == pytest.approx(
        [0.0, 0.5, 0.0]
    )


def test_constant_adjustment_factor_matches_raw_return() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    raw = [10.0, 11.0]
    factor = 7.0
    path = run_continuous_target_path(
        pl.DataFrame({"time": [days[0]], "asset_id": ["a"], "weight": [1.0]}),
        pl.DataFrame(
            {
                "time": days,
                "asset_id": ["a", "a"],
                "total_return_price": [value * factor for value in raw],
            }
        ),
        initial_capital=1_000.0,
        transaction_cost=_cost(),
    )
    assert path.returns.get_column("gross_return").to_list() == pytest.approx(
        [0.0, 0.1]
    )


def test_cash_dividend_is_execution_cash_but_not_a_second_performance_return() -> None:
    days = [date(2024, 1, day) for day in range(2, 5)]
    targets = pl.DataFrame(
        {"time": [days[0]], "asset_id": ["a"], "weight": [1.0]}
    )
    raw_prices = pl.DataFrame(
        {
            "time": days,
            "asset_id": ["a"] * 3,
            "open": [10.0, 9.9, 9.9],
            "close": [10.0, 9.9, 9.9],
        }
    )
    actions = pl.DataFrame(
        {
            "action_id": ["div-a"],
            "asset_id": ["a"],
            "is_implemented": [True],
            "record_date": [days[0]],
            "ex_date": [days[1]],
            "cash_pay_date": [days[2]],
            "share_available_date": [None],
            "cash_dividend_per_share": [0.1],
            "stock_dividend_per_share": [0.0],
        }
    )
    account = run_account_backtest(
        targets,
        raw_prices,
        corporate_action_coverage=pl.DataFrame(
            {"time": days, "is_complete": [True] * 3}
        ),
        corporate_actions=actions,
        config=AccountBacktestConfig(
            capital_mode="compounding",
            initial_capital=1_000.0,
            transaction_cost=_cost(),
        ),
    )
    total_return = pl.DataFrame(
        {
            "time": days,
            "asset_id": ["a"] * 3,
            "total_return_price": [10.0, 10.0, 10.0],
            "execution_total_return_price": [10.0, 10.0, 10.0],
        }
    )
    target = run_continuous_target_path(
        targets,
        total_return,
        initial_capital=1_000.0,
        transaction_cost=_cost(),
    )
    actual = run_actual_performance_path(
        account.fills,
        account.positions,
        total_return,
        initial_capital=1_000.0,
    )
    comparison = compare_portfolio_paths(
        target, actual, account, initial_capital=1_000.0
    )

    assert actual.returns.get_column("gross_return").to_list() == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert account.receivables.get_column("kind").to_list() == [
        "cash_created",
        "cash_paid",
    ]
    assert comparison.reconciliation.get_column("unexplained_gap").to_list() == (
        pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    )
