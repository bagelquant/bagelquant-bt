"""Common-coordinate Alpha IC and saved-fill account turnover."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import polars as pl

from .evaluation import _window_labels
from .factor_analysis import partial_rank_ic
from .horizon import SessionWindow
from .portfolio_mechanics import normalize_forward_returns, session_calendar


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


def common_alpha_ic(
    baseline: pl.DataFrame,
    variant: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    calendar: pl.DataFrame,
    horizon: int = 5,
    available_date: date | None = None,
    batch_sessions: int = 64,
    check_canceled: Callable[[], None] = lambda: None,
) -> pl.DataFrame:
    """Compare rank IC on identical Alpha coordinates and caller-labeled returns.

    No execution shift or price-derived label is inserted. Complete supplied
    daily labels are compounded for the requested horizon, immature windows
    are excluded and incomplete member labels remain unavailable.
    """
    for name, value in (("horizon", horizon), ("batch_sessions", batch_sessions)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for frame in (baseline, variant):
        if not isinstance(frame, pl.DataFrame) or not {
            "time",
            "asset_id",
            "value",
        } <= set(frame.columns):
            raise ValueError("comparison Alpha inputs require time, asset_id, value")
        if frame.select(pl.struct("time", "asset_id").is_duplicated().any()).item():
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
    sessions = session_calendar(calendar, default=common["time"].unique().sort())
    labels = normalize_forward_returns(
        forward_returns, calendar=sessions, available_date=available_date
    )
    factor = common.select(
        pl.col("time").alias("evaluation_date"),
        pl.col("time").alias("execution_date"),
        "asset_id",
        pl.col("baseline").alias("factor"),
    )
    window = SessionWindow("cumulative", f"cumulative_{horizon}d", 1, horizon)
    mature, _ = _window_labels(factor, labels, sessions, window)
    paired = mature.join(
        common.rename({"time": "evaluation_date"}),
        on=["evaluation_date", "asset_id"],
        how="inner",
    )
    rows = []
    for index, ((day,), sample) in enumerate(
        paired.group_by("evaluation_date", maintain_order=True)
    ):
        if index % batch_sessions == 0:
            check_canceled()
        before, after, realized = (
            sample[column].to_numpy()
            for column in ("baseline", "variant", "forward_return")
        )
        left, right = (
            partial_rank_ic(before, realized),
            partial_rank_ic(after, realized),
        )
        conditional = partial_rank_ic(after, realized, before[:, None])
        difference = (
            None
            if left["ic"] is None or right["ic"] is None
            else right["ic"] - left["ic"]
        )
        rows.append(
            {
                "time": day,
                "target_end_date": sample["target_end_date"].max(),
                "available_date": sample["available_date"].max(),
                "baseline_ic": left["ic"],
                "variant_ic": right["ic"],
                "spearman_ic_increment": difference,
                "conditional_ic": conditional["ic"],
                "sample_count": right["sample_count"],
                "reason": conditional["reason"],
                "window_id": window.window_id,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "time": pl.Date,
            "target_end_date": pl.Date,
            "available_date": pl.Date,
            "baseline_ic": pl.Float64,
            "variant_ic": pl.Float64,
            "spearman_ic_increment": pl.Float64,
            "conditional_ic": pl.Float64,
            "sample_count": pl.Int64,
            "reason": pl.String,
            "window_id": pl.String,
        },
    ).sort("time")
