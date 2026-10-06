"""Common-coordinate prediction IC and saved-fill account turnover."""

from __future__ import annotations

import polars as pl
from bagelquant_core import Domain, Node

from .factor_analysis import partial_rank_ic
from .horizon import SessionWindow, session_window_forward_returns
from .policy import ExecutionPolicy


def account_fill_turnover(
    fills: pl.DataFrame, account_value: pl.DataFrame, *, initial_capital: float
) -> pl.DataFrame:
    """Gross absolute filled notional divided by strictly previous close equity."""
    previous = account_value.sort("time").select(
        "time",
        pl.col("equity").shift(1).fill_null(initial_capital).alias("previous_equity"),
    )
    traded = fills.group_by("time").agg(
        pl.col("notional").abs().sum().alias("traded_notional")
    )
    return (
        previous.join(traded, on="time", how="left")
        .with_columns(pl.col("traded_notional").fill_null(0))
        .with_columns(
            pl.when(pl.col("previous_equity") > 0)
            .then(pl.col("traded_notional") / pl.col("previous_equity"))
            .otherwise(None)
            .alias("gross_traded_fraction")
        )
    )


def common_prediction_ic(
    baseline: pl.DataFrame,
    variant: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    calendar: pl.DataFrame,
    horizon: int = 5,
    session_lag: int = 1,
    batch_sessions: int = 64,
    check_canceled=lambda: None,
) -> pl.DataFrame:
    """Compute baseline, variant and conditional rank IC on identical asset keys.

    Each bounded date batch constructs forward labels once. Only daily statistics
    are returned; no second wide feature or asset/horizon label matrix is stored.
    """
    if horizon < 1 or session_lag < 1 or batch_sessions < 1:
        raise ValueError("horizon, session lag and batch size must be positive")
    for frame in (baseline, variant):
        if frame.select("time", "asset_id").is_duplicated().any():
            raise ValueError("comparison input keys must be unique")
    common = (
        baseline.select("time", "asset_id", pl.col("value").alias("baseline"))
        .join(
            variant.select("time", "asset_id", pl.col("value").alias("variant")),
            on=["time", "asset_id"],
            how="inner",
        )
        .filter(pl.col("baseline").is_finite() & pl.col("variant").is_finite())
        .sort("time", "asset_id")
    )
    dates = common["time"].unique().sort().to_list()
    rows = []
    from datetime import date

    schema = {
        "time": pl.Date,
        "target_end_date": pl.Date,
        "baseline_ic": pl.Float64,
        "variant_ic": pl.Float64,
        "spearman_ic_increment": pl.Float64,
        "conditional_ic": pl.Float64,
        "sample_count": pl.Int64,
        "reason": pl.String,
        "window_id": pl.String,
    }
    for offset in range(0, len(dates), batch_sessions):
        check_canceled()
        batch = common.filter(
            pl.col("time").is_between(
                dates[offset], dates[min(offset + batch_sessions, len(dates)) - 1]
            )
        )
        domain = Domain(
            calendar=calendar["time"],
            universe=batch.select("time", "asset_id").with_columns(
                pl.lit(True).alias("active")
            ),
        )
        panel = Node.from_domain(
            batch.select("time", "asset_id", pl.col("baseline").alias("value")), domain
        , value_type="prediction")
        scheduled = ExecutionPolicy(
            "next_open", lag_sessions=session_lag
        ).schedule_prediction(panel, calendar)
        window = SessionWindow("cumulative", f"cumulative_{horizon}d", 1, horizon)
        labels, _ = session_window_forward_returns(
            scheduled, prices, windows=[window], calendar=calendar
        )
        paired = labels.join(
            batch.rename({"time": "evaluation_date"}),
            on=["evaluation_date", "asset_id"],
            how="inner",
        )
        for (day,), sample in paired.group_by("evaluation_date", maintain_order=True):
            before, after, returns = (
                sample[column].to_numpy()
                for column in ("baseline", "variant", "forward_return")
            )
            left, right = (
                partial_rank_ic(before, returns),
                partial_rank_ic(after, returns),
            )
            conditional = partial_rank_ic(after, returns, before[:, None])
            difference = (
                None
                if left["ic"] is None or right["ic"] is None
                else right["ic"] - left["ic"]
            )
            ending = sample["target_end_date"].max()
            assert isinstance(ending, date)
            rows.append(
                {
                    "time": day,
                    "target_end_date": ending,
                    "baseline_ic": left["ic"],
                    "variant_ic": right["ic"],
                    "spearman_ic_increment": difference,
                    "conditional_ic": conditional["ic"],
                    "sample_count": right["sample_count"],
                    "reason": conditional["reason"],
                    "window_id": window.window_id,
                }
            )
    return pl.DataFrame(rows, schema=schema).sort("time")
