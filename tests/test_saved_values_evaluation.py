from datetime import date, timedelta

import polars as pl
import pytest
from bagelquant_core import Domain, Node
from polars.testing import assert_frame_equal

from bagelquant_bt import (
    AccountBacktestConfig,
    BacktestConfig,
    ExecutionPolicy,
    evaluate_portfolio_targets,
    implementation_stress_scenarios,
    run_daily_prediction_sections,
)


@pytest.mark.parametrize("scenario_index", range(9))
@pytest.mark.parametrize("cutoff_index", [5, 6, 8])
def test_every_stress_scenario_has_independent_full_and_pending_plan_continuation(
    scenario_index, cutoff_index,
):
    days = [date(2024, 1, 2) + timedelta(days=i) for i in range(18)]
    calendar = pl.DataFrame({"time": days})
    prices = pl.DataFrame(
        [
            {
                "time": day,
                "asset_id": asset,
                "open": 10.0 + i * 0.05,
                "close": 10.1 + i * 0.05,
            }
            for i, day in enumerate(days)
            for asset in ("a", "b")
        ]
    )
    decisions = calendar.with_columns(
        pl.when(pl.col("time").is_in([days[0], days[5], days[10]]))
        .then(pl.lit("rebalance"))
        .otherwise(pl.lit("hold"))
        .alias("status"),
        pl.lit(None, dtype=pl.String).alias("reason"),
    )
    weights = pl.DataFrame(
        [
            {"time": day, "asset_id": asset, "value": weight}
            for day, sleeve in [
                (days[0], (1.0, 0.0)),
                (days[5], (0.0, 1.0)),
                (days[10], (0.0, 0.0)),
            ]
            for asset, weight in zip(("a", "b"), sleeve, strict=True)
        ]
    )
    original = AccountBacktestConfig(initial_capital=10000.0)
    name, config, lag = list(implementation_stress_scenarios(original, session_lag=1))[
        scenario_index
    ]
    assert name and config.transaction_cost.min_fee == original.transaction_cost.min_fee
    assert (
        config.transaction_cost.stamp_tax_rate
        == original.transaction_cost.stamp_tax_rate
    )
    kwargs = {
        "config": config,
        "session_lag": lag,
        "corporate_action_coverage": calendar.with_columns(
            pl.lit(True).alias("is_complete")
        ),
    }
    full = evaluate_portfolio_targets(
        weights, decisions, prices, calendar=calendar, **kwargs
    )
    cutoff = days[cutoff_index]
    prefix = evaluate_portfolio_targets(
        weights.filter(pl.col("time") <= cutoff),
        decisions.filter(pl.col("time") <= cutoff),
        prices.filter(pl.col("time") <= cutoff),
        calendar=calendar,
        **kwargs,
    )
    suffix = evaluate_portfolio_targets(
        weights,
        decisions,
        prices,
        calendar=calendar,
        checkpoint=prefix.account.final_checkpoint,
        initial_target_position_plans=prefix.target_position_plans.filter(
            pl.col("execution_date") > cutoff
        ),
        historical_positions=prefix.account.positions,
        historical_sessions=prefix.account.account_value,
        **kwargs,
    )
    for field in (
        "account_value",
        "performance",
        "positions",
        "cash",
        "orders",
        "fills",
        "receivables",
    ):
        assert_frame_equal(
            pl.concat(
                [getattr(prefix.account, field), getattr(suffix.account, field)],
                how="vertical_relaxed",
            ),
            getattr(full.account, field),
        )


def test_saved_targets_continue_pending_decision_and_full_exit():
    days = [date(2024, 1, 2) + timedelta(days=i) for i in range(12)]
    calendar = pl.DataFrame({"time": days})
    prices = pl.DataFrame(
        [
            {
                "time": day,
                "asset_id": asset,
                "open": 10.0 + i / 10,
                "close": 10.05 + i / 10,
            }
            for i, day in enumerate(days)
            for asset in ["a", "b"]
        ]
    )
    decisions = calendar.with_columns(
        pl.when(pl.col("time").is_in([days[0], days[5], days[10]]))
        .then(pl.lit("rebalance"))
        .otherwise(pl.lit("hold"))
        .alias("status"),
        pl.lit(None, dtype=pl.String).alias("reason"),
    )
    weights = pl.DataFrame(
        [
            {"time": day, "asset_id": asset, "value": weight}
            for day, sleeve in [
                (days[0], (1.0, 0.0)),
                (days[5], (0.0, 1.0)),
                (days[10], (0.0, 0.0)),
            ]
            for asset, weight in zip(["a", "b"], sleeve, strict=True)
        ]
    )
    kwargs = dict(
        config=AccountBacktestConfig(initial_capital=10000),
        corporate_action_coverage=calendar.with_columns(
            pl.lit(True).alias("is_complete")
        ),
    )
    full = evaluate_portfolio_targets(
        weights, decisions, prices, calendar=calendar, **kwargs
    )
    # End exactly on a decision: its next-session plan must survive the split.
    end = days[5]
    prefix = evaluate_portfolio_targets(
        weights.filter(pl.col("time") <= end),
        decisions.filter(pl.col("time") <= end),
        prices.filter(pl.col("time") <= end),
        calendar=calendar,
        **kwargs,
    )
    suffix = evaluate_portfolio_targets(
        weights,
        decisions,
        prices,
        calendar=calendar,
        checkpoint=prefix.account.final_checkpoint,
        initial_target_position_plans=prefix.target_position_plans.filter(
            pl.col("execution_date") > end
        ),
        **kwargs,
    )
    for name in (
        "account_value",
        "performance",
        "positions",
        "fills",
        "orders",
        "cash",
    ):
        actual = pl.concat(
            [getattr(prefix.account, name), getattr(suffix.account, name)],
            how="vertical_relaxed",
        )
        assert_frame_equal(actual, getattr(full.account, name))
    assert full.account.positions.filter(pl.col("time") == days[-1]).is_empty()


def test_saved_targets_reject_unknown_future_execution_session():
    days = [date(2024,1,2),date(2024,1,3)]
    weights = pl.DataFrame({'time':[days[0]],'asset_id':['a'],'value':[1.]})
    decisions = pl.DataFrame(
        {'time':days,'status':['rebalance','hold'],'reason':[None,None]},
        schema_overrides={'reason':pl.String})
    prices = pl.DataFrame(
        {'time':days,'asset_id':['a','a'],'open':[10.,10.],'close':[10.,10.]}
    )
    from bagelquant_bt.exceptions import InputValidationError
    with pytest.raises(InputValidationError,match='known execution sessions'):
        evaluate_portfolio_targets(weights,decisions,prices,calendar=pl.DataFrame({'time':days}),
            corporate_action_coverage=pl.DataFrame({'time':days,'is_complete':[True,True]}),
            config=AccountBacktestConfig(),session_lag=2)


def test_stress_accounts_share_prepared_prices_without_sharing_state(monkeypatch):
    import bagelquant_bt.account as account_module
    from bagelquant_bt import prepare_account_market_data
    days = [date(2024,1,2)+timedelta(days=index) for index in range(8)]
    calendar = pl.DataFrame({'time':days})
    prices = calendar.with_columns(
        pl.lit('a').alias('asset_id'),pl.lit(10.).alias('open'),
        pl.Series('close',[10.+index*.1 for index in range(8)]),
    )
    weights = pl.DataFrame({'time':[days[0]],'asset_id':['a'],'value':[1.]})
    decisions = calendar.with_columns(
        pl.when(pl.col('time')==days[0]).then(pl.lit('rebalance'))
        .otherwise(pl.lit('hold')).alias('status'),
        pl.lit(None,dtype=pl.String).alias('reason'),
    )
    coverage = calendar.with_columns(pl.lit(True).alias('is_complete'))
    original = account_module._price_lookup
    calls = []
    def lookup(frame):
        calls.append(frame.height)
        return original(frame)
    monkeypatch.setattr(account_module,'_price_lookup',lookup)
    availability = calendar.with_columns(pl.lit('a').alias('asset_id'),
        (pl.col('time') != days[1]).alias('can_buy'),pl.lit(True).alias('can_sell'),
        pl.lit('blocked').alias('reason'))
    availability_calls = []
    original_availability = account_module._availability_lookup
    def availability_lookup(frame):
        availability_calls.append(frame.height if frame is not None else None)
        return original_availability(frame)
    monkeypatch.setattr(account_module,'_availability_lookup',availability_lookup)
    prepared = prepare_account_market_data(prices,execution_availability=availability)
    scenarios = list(implementation_stress_scenarios(
        AccountBacktestConfig(initial_capital=10000.),session_lag=1,
    ))
    results = []
    for _name,config,lag in scenarios:
        result = evaluate_portfolio_targets(
            weights,decisions,prepared,calendar=calendar,
            corporate_action_coverage=coverage,config=config,session_lag=lag,
        ).account
        results.append(result)
    assert calls == [prices.height]
    assert availability_calls == [availability.height]
    from bagelquant_bt.exceptions import InputValidationError
    with pytest.raises(InputValidationError,match='already includes'):
        evaluate_portfolio_targets(weights,decisions,prepared,calendar=calendar,
            corporate_action_coverage=coverage,config=scenarios[0][1],
            execution_availability=availability)
    monkeypatch.setattr(account_module,'_price_lookup',original)
    for result,(_,config,lag) in zip(results,scenarios,strict=True):
        reference = evaluate_portfolio_targets(
            weights,decisions,prices,calendar=calendar,
            corporate_action_coverage=coverage,config=config,session_lag=lag,
            execution_availability=availability,
        ).account
        for name in ('account_value','positions','cash','orders','fills'):
            assert_frame_equal(getattr(result,name),getattr(reference,name))


def test_horizon_append_reuses_prefix_and_matches_full_newly_matured_labels(
    monkeypatch,
):
    days = [date(2023, 1, 2) + timedelta(days=i) for i in range(380)]
    assets = [f"a{i:02}" for i in range(12)]
    values = pl.DataFrame(
        [
            {"time": day, "asset_id": asset, "value": float((j * 3 + i // 7) % 13)}
            for i, day in enumerate(days)
            for j, asset in enumerate(assets)
        ]
    )
    prices = pl.DataFrame(
        [
            {"time": day, "asset_id": asset, "price": 100.0 + i * (j + 1) / 100}
            for i, day in enumerate(days)
            for j, asset in enumerate(assets)
        ]
    )

    def run(end, **kwargs):
        calendar = pl.DataFrame({"time": days[:end]})
        panel = Node.from_domain(
            values.filter(pl.col("time") <= days[end - 1]),
            Domain(calendar=calendar["time"], universe=assets),
         value_type="prediction")
        signals = ExecutionPolicy("next_open").schedule_prediction(panel, calendar)
        return run_daily_prediction_sections(
            signals,
            prices.filter(pl.col("time") <= days[end - 1]),
            config=BacktestConfig(initial_capital=10000),
            calendar=calendar,
            components=[
                "horizons",
                "persistence",
                "rolling_ic",
                "book_tail",
                "quantiles",
                "lead_lag",
                "alpha_return",
                "turnover",
            ],
            **kwargs,
        )

    old, _ = run(360)
    import bagelquant_bt.horizon as module

    observed_rows = []
    original = module._prepare_daily_weights

    def observed(context, **kwargs):
        observed_rows.append(context.factor.height)
        return original(context, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_prepare_daily_weights", observed)
        incremental, sliced = run(380, prior_frames=old, prior_through=days[359])
    assert len(observed_rows) == 1
    assert max(observed_rows) < values.height
    observed_rows.clear()
    with monkeypatch.context() as patch:
        patch.setattr(module, "_prepare_daily_weights", observed)
        full, complete = run(380)
    assert len(observed_rows) == 1
    for name in ("daily_book_lead_lag_returns", "daily_alpha_return_lag_returns"):
        assert_frame_equal(
            incremental[name],
            full[name],
            check_exact=False,
            abs_tol=1e-10,
            rel_tol=1e-10,
        )
    assert incremental.keys() == full.keys()
    for name in full:
        actual, expected = incremental[name], full[name]
        assert actual.height == expected.height, (name, actual.shape, expected.shape)
        sort = [
            key
            for key in (
                "evaluation_date",
                "time",
                "window_id",
                "horizon_sessions",
                "quantile",
                "metric",
            )
            if key in expected.columns
        ]
        assert_frame_equal(
            actual.sort(sort) if sort else actual,
            expected.sort(sort) if sort else expected,
            check_exact=False,
            abs_tol=1e-10,
        )
    assert sliced.max_window_forward_rows < complete.max_window_forward_rows
    # Adding chapters after a short first evaluation needs their entire history.
    newly_selected, _ = run(
        380,
        prior_frames={
            "daily_signal_autocorrelation": old["daily_signal_autocorrelation"]
        },
        prior_through=days[359],
    )
    for name, expected in full.items():
        sort = [
            key
            for key in (
                "evaluation_date",
                "time",
                "window_id",
                "horizon_sessions",
                "quantile",
                "metric",
            )
            if key in expected.columns
        ]
        assert_frame_equal(
            newly_selected[name].sort(sort) if sort else newly_selected[name],
            expected.sort(sort) if sort else expected,
            check_exact=False,
            abs_tol=1e-10,
        )


def test_account_output_chunks_preserve_schema_order_and_promote_nulls(monkeypatch):
    import bagelquant_bt.account as module

    monkeypatch.setattr(module, "_ACCOUNT_ROW_BATCH_SIZE", 2)
    rows = [
        {"time": date(2024, 1, 1), "quantity": 0, "price": None},
        {"time": date(2024, 1, 2), "quantity": 1, "price": None},
        {"time": date(2024, 1, 3), "quantity": 2, "price": 10.0},
        {"time": date(2024, 1, 4), "quantity": 3, "price": 11.0,
         "reason": "corporate action"},
        {"time": date(2024, 1, 5), "quantity": 0, "price": 12.0},
    ]
    output = module._AccountRows()
    for row in rows:
        output.append(row)
        assert len(output._buffer) < 2
    assert_frame_equal(
        module._rows_frame(output), pl.DataFrame(rows, infer_schema_length=None)
    )


def test_chunked_account_matches_full_rows_and_checkpoint(monkeypatch):
    from dataclasses import fields

    import bagelquant_bt.account as module

    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(35)]
    calendar = pl.DataFrame({"time": days})
    prices = calendar.join(pl.DataFrame({"asset_id": ["a", "b"]}), how="cross")
    prices = prices.with_columns(
        pl.lit(10.0).alias("open"), pl.lit(10.1).alias("close")
    )
    weights = pl.DataFrame({
        "time": [days[0], days[0], days[10], days[10], days[20], days[20]],
        "asset_id": ["a", "b"] * 3, "value": [1., 0., .5, .5, 0., 0.],
    })
    decisions = calendar.with_columns(
        pl.when(pl.col("time").is_in(weights["time"].unique().implode()))
        .then(pl.lit("rebalance")).otherwise(pl.lit("hold")).alias("status"),
        pl.lit(None, dtype=pl.String).alias("reason"),
    )

    def run():
        return evaluate_portfolio_targets(
            weights, decisions, prices, calendar=calendar,
            corporate_action_coverage=calendar.with_columns(
                pl.lit(True).alias("is_complete")
            ), config=AccountBacktestConfig(initial_capital=10000),
        ).account

    monkeypatch.setattr(module, "_ACCOUNT_ROW_BATCH_SIZE", 100000)
    expected = run()
    monkeypatch.setattr(module, "_ACCOUNT_ROW_BATCH_SIZE", 3)
    actual = run()
    for field in fields(actual):
        if isinstance(getattr(actual, field.name), pl.DataFrame):
            assert_frame_equal(
                getattr(actual, field.name), getattr(expected, field.name)
            )
    for field in fields(actual.final_checkpoint):
        value = getattr(actual.final_checkpoint, field.name)
        reference = getattr(expected.final_checkpoint, field.name)
        if isinstance(value, pl.DataFrame):
            assert_frame_equal(value, reference)
        else:
            assert value == reference


def test_turnover_checkpoint_retains_long_blocks_and_absent_price_targets():
    from bagelquant_bt.engine import _sparse_executed_turnover

    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    weights = pl.DataFrame(
        [
            {"time": day, "asset_id": asset, "weight": float((i // 5 + j) % 2)}
            for i, day in enumerate(days)
            for j, asset in enumerate(["a", "b"])
        ]
    )
    prices = pl.DataFrame(
        [
            {"time": day, "asset_id": asset, "price": 10.0}
            for i, day in enumerate(days)
            for asset in ["a", "b"]
            if not (asset == "b" and 8 <= i <= 22)
        ]
    )
    returns = (
        prices.filter(pl.col("time") < days[-1])
        .select("time", "asset_id")
        .with_columns(pl.lit(0.0).alias("forward_return"))
    )
    availability = pl.DataFrame(
        [
            {
                "time": day,
                "asset_id": asset,
                "can_buy": i > 23,
                "can_sell": i > 23,
                "reason": "blocked",
            }
            for i, day in enumerate(days)
            for asset in ["a", "b"]
        ]
    )
    args = dict(
        execution_availability=availability, retry_blocked=True, return_checkpoint=True
    )
    full, final = _sparse_executed_turnover(weights, prices, returns, **args)
    old, checkpoint = _sparse_executed_turnover(
        weights.filter(pl.col("time") <= days[19]),
        prices.filter(pl.col("time") <= days[19]),
        returns.filter(pl.col("time") < days[19]),
        **args,
    )
    suffix, saved = _sparse_executed_turnover(
        weights, prices, returns, checkpoint=checkpoint, **args
    )
    assert_frame_equal(pl.concat([old, suffix]), full)
    assert_frame_equal(saved, final)
