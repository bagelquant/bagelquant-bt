"""Caller-aligned portfolio construction and fractional position mechanics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import polars as pl
from bagelquant_core import Domain, Node

from .exceptions import InputValidationError
from .horizon import centered_rank_book_weights, gross_one_tail_weights


@dataclass(frozen=True, slots=True)
class WeightStateCheckpoint:
    """Independent Gross/Net holdings after the last mature return interval."""

    time: date
    interval_end: date
    available_date: date
    gross_value: float
    net_value: float
    weights: pl.DataFrame


def node_values(node: Node, *, kind: Literal["alpha", "weights"]) -> pl.DataFrame:
    """Collect an explicitly typed saved value without computing its graph."""

    allowed = {"numeric", "prediction"} if kind == "alpha" else {"weights"}
    if not isinstance(node, Node) or node.value_type not in allowed:
        raise TypeError(f"{kind} must be a saved Node with value_type in {allowed}")
    frame = node.collect(dense=False).select("time", "asset_id", "value")
    if (
        kind == "weights"
        and frame.filter(
            pl.col("value").is_null() | ~pl.col("value").is_finite()
        ).height
    ):
        raise InputValidationError("weights must be finite complete snapshots")
    return frame.filter(pl.col("value").is_finite()).sort("time", "asset_id")


def session_calendar(
    calendar: pl.DataFrame | None, *, default: pl.Series
) -> pl.DataFrame:
    """Validate the explicit ordered session coordinate system."""

    frame = pl.DataFrame({"time": default}) if calendar is None else calendar
    if not isinstance(frame, pl.DataFrame) or "time" not in frame.columns:
        raise InputValidationError("calendar requires a time column")
    result = frame.select(pl.col("time").cast(pl.Date, strict=False))
    if result["time"].null_count() or result["time"].n_unique() != result.height:
        raise InputValidationError("calendar dates must be unique and non-null")
    return result.sort("time")


def normalize_forward_returns(
    forward_returns: pl.DataFrame,
    *,
    calendar: pl.DataFrame | None = None,
    available_date: date | None = None,
) -> pl.DataFrame:
    """Validate caller-labeled returns; no price inference or hidden time shift.

    ``time`` is the caller's start label. ``interval_end`` identifies the actual
    economic endpoint, ``interval_start`` its actual economic start, and
    ``available_date`` records when that label is known.
    Null returns are mature coverage gaps only when both dates are at or before
    the explicitly supplied cutoff. Without a cutoff, the latest declared
    availability is used, never today's date.
    """

    required = {
        "time",
        "asset_id",
        "forward_return",
        "interval_start",
        "interval_end",
        "available_date",
    }
    if not isinstance(forward_returns, pl.DataFrame) or not required <= set(
        forward_returns.columns
    ):
        raise InputValidationError(
            "forward_returns requires time, asset_id, forward_return, "
            "interval_start, interval_end and available_date columns"
        )
    frame = forward_returns.select(
        pl.col("time").cast(pl.Date, strict=False),
        pl.col("asset_id").cast(pl.String),
        pl.col("forward_return").cast(pl.Float64, strict=False).fill_nan(None),
        pl.col("interval_start").cast(pl.Date, strict=False),
        pl.col("interval_end").cast(pl.Date, strict=False),
        pl.col("available_date").cast(pl.Date, strict=False),
    )
    if frame.filter(
        pl.any_horizontal(
            pl.col(name).is_null()
            for name in (
                "time",
                "asset_id",
                "interval_start",
                "interval_end",
                "available_date",
            )
        )
        | (pl.col("asset_id").str.len_chars() == 0)
        | (pl.col("interval_start") < pl.col("time"))
        | (pl.col("interval_end") <= pl.col("interval_start"))
        | (pl.col("available_date") < pl.col("interval_end"))
        | (
            pl.col("forward_return").is_not_null()
            & ~pl.col("forward_return").is_finite()
        )
        | (pl.col("forward_return") < -1.0)
    ).height:
        raise InputValidationError("forward_returns contains invalid values or dates")
    if frame.select(pl.struct("time", "asset_id").is_duplicated().any()).item():
        raise InputValidationError("forward_returns must be unique by (time, asset_id)")
    if (
        frame.group_by("time")
        .agg(pl.col("interval_start").n_unique(), pl.col("interval_end").n_unique())
        .filter((pl.col("interval_end") != 1) | (pl.col("interval_start") != 1))
        .height
    ):
        raise InputValidationError(
            "each alignment label requires one economic interval"
        )
    timing = (
        frame.select("time", "interval_start", "interval_end").unique().sort("time")
    )
    if timing.filter(pl.col("interval_start") < pl.col("interval_end").shift(1)).height:
        raise InputValidationError("economic forward-return intervals must not overlap")
    if (
        calendar is not None
        and frame.select("time")
        .unique()
        .join(calendar.select("time"), on="time", how="anti")
        .height
    ):
        raise InputValidationError(
            "forward-return start labels are absent from calendar"
        )
    cutoff = (
        available_date if available_date is not None else frame["available_date"].max()
    )
    if cutoff is not None and not isinstance(cutoff, date):
        raise InputValidationError("available_date cutoff must be a date")
    return frame.with_columns(
        ((pl.col("interval_end") <= cutoff) & (pl.col("available_date") <= cutoff))
        .fill_null(False)
        .alias("is_mature")
    ).sort("time", "asset_id")


def select_rebalance_snapshots(
    values: pl.DataFrame,
    *,
    calendar: pl.DataFrame,
    every: int,
    anchor: date | None,
) -> pl.DataFrame:
    """Select whole latest snapshots on an anchored session cadence."""

    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise InputValidationError("every must be a positive integer")
    if values.is_empty():
        return values
    start = anchor if anchor is not None else values["time"].min()
    sessions = calendar.filter(pl.col("time") >= start).with_row_index("_ordinal")
    requested = sessions.filter(pl.col("_ordinal") % every == 0).select("time")
    snapshots = (
        values.select(pl.col("time").alias("source_time")).unique().sort("source_time")
    )
    schedule = requested.join_asof(
        snapshots, left_on="time", right_on="source_time", strategy="backward"
    ).drop_nulls("source_time")
    return (
        schedule.join(
            values.rename({"time": "source_time"}), on="source_time", how="inner"
        )
        .select("time", "asset_id", "value")
        .sort("time", "asset_id")
    )


def build_alpha_weights(
    alpha: Node,
    *,
    method: Literal["book", "spread"] = "book",
    quantiles: int = 10,
    every: int = 1,
    calendar: pl.DataFrame | None = None,
    anchor: date | None = None,
) -> Node:
    """Build net-zero, gross-one Book or equal-weight tail Spread snapshots.

    Book centers average percentile ranks and divides by their absolute sum.
    Spread is long one half in the highest quantile and short one half in the
    lowest. Degenerate cross-sections remain explicitly unavailable (null),
    rather than silently becoming an order or a smaller portfolio.
    """

    values = node_values(alpha, kind="alpha")
    if method not in {"book", "spread"}:
        raise InputValidationError("method must be 'book' or 'spread'")
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or quantiles < 2:
        raise InputValidationError("quantiles must be an integer of at least two")
    if every != 1 and (calendar is None or anchor is None):
        raise InputValidationError(
            "non-daily construction requires calendar and anchor"
        )
    sessions = session_calendar(calendar, default=alpha.domain.times)
    values = select_rebalance_snapshots(
        values, calendar=sessions, every=every, anchor=anchor
    )
    factor = values.select(
        pl.col("time").alias("evaluation_date"),
        pl.col("time").alias("execution_date"),
        "asset_id",
        pl.col("value").alias("factor"),
    )
    built = (
        centered_rank_book_weights(factor)
        if method == "book"
        else gross_one_tail_weights(factor, quantiles=quantiles)
    )
    name = "book_weight" if method == "book" else "tail_weight"
    frame = built.select(
        pl.col("evaluation_date").alias("time"), "asset_id", pl.col(name).alias("value")
    )
    domain = Domain(calendar=sessions["time"], universe=alpha.domain.asset_ids)
    return Node.from_domain(
        frame,
        domain,
        value_type="weights",
        name=f"{method}_weights",
        metadata={"method": method, "quantiles": quantiles, "every": every},
    )


def simulate_weight_returns(
    weights: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    decisions: pl.DataFrame | None,
    cost_rate: float,
    checkpoint: WeightStateCheckpoint | None = None,
    include_holdings: bool = True,
    check_canceled: Callable[[], None] | None = None,
) -> tuple[dict[str, pl.DataFrame], WeightStateCheckpoint | None]:
    """Drift independent Gross/Net holdings and charge full Net-pretrade L1.

    The per-session loop is the portfolio state machine; cross-sectional
    return, cost and drift calculations use fixed-order NumPy vectors.
    """

    if not math.isfinite(cost_rate) or cost_rate < 0:
        raise InputValidationError("cost_rate must be finite and nonnegative")
    if decisions is not None:
        if not {"time", "status"} <= set(decisions.columns):
            raise InputValidationError("decisions requires time and status")
        if (
            decisions["time"].null_count()
            or decisions["time"].n_unique() != decisions.height
        ):
            raise InputValidationError("decision dates must be unique and non-null")
        if decisions.filter(
            ~pl.col("status").is_in(["rebalance", "hold", "unavailable"])
            | pl.col("status").is_null()
        ).height:
            raise InputValidationError("unknown decision status")
        rebalances = set(decisions.filter(pl.col("status") == "rebalance")["time"])
        if set(weights["time"]) != rebalances:
            raise InputValidationError(
                "weight snapshots must exactly match rebalance decisions"
            )
    else:
        rebalances = set(weights["time"])
    if checkpoint is not None and not isinstance(checkpoint, WeightStateCheckpoint):
        raise TypeError("checkpoint must be a WeightStateCheckpoint")
    assets = sorted(
        set(weights["asset_id"])
        | set(labels["asset_id"])
        | (set(checkpoint.weights["asset_id"]) if checkpoint else set())
    )
    asset_indices = {asset: index for index, asset in enumerate(assets)}
    count = len(assets)
    gross = np.zeros(count)
    net = np.zeros(count)
    gross_value = net_value = 1.0
    if checkpoint is not None:
        if not {"asset_id", "gross_weight", "net_weight"} <= set(
            checkpoint.weights.columns
        ):
            raise InputValidationError("invalid checkpoint weights")
        if (
            checkpoint.weights["asset_id"].n_unique() != checkpoint.weights.height
            or checkpoint.weights["asset_id"].null_count()
            or checkpoint.weights.select(
                pl.any_horizontal(
                    ~pl.col(name).is_finite() | pl.col(name).is_null()
                    for name in ("gross_weight", "net_weight")
                ).any()
            ).item()
            or not all(
                math.isfinite(value) and value > 0
                for value in (checkpoint.gross_value, checkpoint.net_value)
            )
        ):
            raise InputValidationError("checkpoint state must be finite and unique")
        indices = np.array([asset_indices[a] for a in checkpoint.weights["asset_id"]])
        gross[indices] = checkpoint.weights["gross_weight"].to_numpy()
        net[indices] = checkpoint.weights["net_weight"].to_numpy()
        gross_value, net_value = checkpoint.gross_value, checkpoint.net_value
    targets = {}
    indexed_weights = weights.with_columns(
        pl.col("asset_id")
        .replace_strict(asset_indices, return_dtype=pl.Int64)
        .alias("_asset_index")
    )
    for snapshot in indexed_weights.partition_by("time", maintain_order=True):
        values = np.zeros(count)
        indices = snapshot["_asset_index"].to_numpy()
        values[indices] = snapshot["value"].to_numpy()
        targets[snapshot["time"][0]] = values
    return_rows, turnover_rows, value_rows, coverage_rows, holdings = [], [], [], [], []
    final_checkpoint = checkpoint
    active = checkpoint is not None
    indexed_labels = labels.with_columns(
        pl.col("asset_id")
        .replace_strict(asset_indices, return_dtype=pl.Int64)
        .alias("_asset_index")
    )
    for panel in indexed_labels.partition_by("time", maintain_order=True):
        if check_canceled:
            check_canceled()
        day = panel["time"][0]
        if checkpoint is not None and day <= checkpoint.time:
            continue
        # An immature interval is never booked as a zero asset return.
        if not panel["is_mature"].all():
            break
        if day in rebalances:
            desired = targets[day]
            traded = float(np.abs(desired - net).sum())
            gross, net = desired.copy(), desired.copy()
            active = True
            initial_rebalance = final_checkpoint is None and not return_rows
        else:
            traded, initial_rebalance = 0.0, False
        if not active:
            continue
        asset_return = np.zeros(count)
        observed = np.zeros(count, dtype=bool)
        indices = panel["_asset_index"].to_numpy()
        asset_return[indices] = panel["forward_return"].fill_null(0.0).to_numpy()
        observed[indices] = panel["forward_return"].is_not_null().to_numpy()
        expected = (gross != 0) | (net != 0)
        expected_count = int(expected.sum())
        observed_count = int((expected & observed).sum())
        gross_return = float(gross @ asset_return)
        net_pre_cost_return = float(net @ asset_return)
        cost_return = cost_rate * traded
        net_return = net_pre_cost_return - cost_return
        if 1 + gross_return <= 0 or 1 + net_return <= 0:
            raise InputValidationError(f"nonpositive portfolio wealth at {day}")
        interval_end = panel["interval_end"][0]
        interval_start = panel["interval_start"][0]
        availability = panel["available_date"].max()
        if final_checkpoint is not None:
            availability = max(availability, final_checkpoint.available_date)
        if (
            final_checkpoint is not None
            and interval_start != final_checkpoint.interval_end
        ):
            raise InputValidationError(
                f"portfolio returns require contiguous economic intervals at {day}"
            )
        return_rows.append(
            {
                "time": day,
                "interval_start": interval_start,
                "interval_end": interval_end,
                "available_date": availability,
                "gross_return": gross_return,
                "net_pre_cost_return": net_pre_cost_return,
                "cost_return": cost_return,
                "net_return": net_return,
            }
        )
        turnover_rows.append(
            {
                "time": day,
                "interval_start": interval_start,
                "interval_end": interval_end,
                "available_date": availability,
                "turnover": traded,
                "requested_turnover": traded,
                "executed_turnover": traded,
                "is_initial_rebalance": initial_rebalance,
            }
        )
        coverage_rows.append(
            {
                "time": day,
                "interval_end": interval_end,
                "available_date": availability,
                "expected_count": expected_count,
                "observed_count": observed_count,
                "coverage_ratio": observed_count / expected_count
                if expected_count
                else 1.0,
            }
        )
        gross_value *= 1 + gross_return
        net_value *= 1 + net_return
        gross *= (1 + asset_return) / (1 + gross_return)
        net *= (1 + asset_return) / (1 + net_return)
        value_rows.append(
            {
                "time": day,
                "interval_end": interval_end,
                "available_date": availability,
                "gross_value": gross_value,
                "net_value": net_value,
            }
        )
        selected = (gross != 0) | (net != 0)
        if include_holdings:
            holdings.append(
                pl.DataFrame(
                    {
                        "time": [day] * int(selected.sum()),
                        "interval_end": [interval_end] * int(selected.sum()),
                        "available_date": [availability] * int(selected.sum()),
                        "asset_id": np.array(assets)[selected],
                        "gross_weight": gross[selected],
                        "net_weight": net[selected],
                    },
                    schema={
                        "time": pl.Date,
                        "interval_end": pl.Date,
                        "available_date": pl.Date,
                        "asset_id": pl.String,
                        "gross_weight": pl.Float64,
                        "net_weight": pl.Float64,
                    },
                )
            )
        final_checkpoint = WeightStateCheckpoint(
            time=day,
            interval_end=interval_end,
            available_date=availability,
            gross_value=gross_value,
            net_value=net_value,
            weights=pl.DataFrame(
                {"asset_id": assets, "gross_weight": gross, "net_weight": net}
            ),
        )
    schemas = {
        "returns": {
            "time": pl.Date,
            "interval_start": pl.Date,
            "interval_end": pl.Date,
            "available_date": pl.Date,
            "gross_return": pl.Float64,
            "net_pre_cost_return": pl.Float64,
            "cost_return": pl.Float64,
            "net_return": pl.Float64,
        },
        "turnover": {
            "time": pl.Date,
            "interval_start": pl.Date,
            "interval_end": pl.Date,
            "available_date": pl.Date,
            "turnover": pl.Float64,
            "requested_turnover": pl.Float64,
            "executed_turnover": pl.Float64,
            "is_initial_rebalance": pl.Boolean,
        },
        "value": {
            "time": pl.Date,
            "interval_end": pl.Date,
            "available_date": pl.Date,
            "gross_value": pl.Float64,
            "net_value": pl.Float64,
        },
        "coverage": {
            "time": pl.Date,
            "interval_end": pl.Date,
            "available_date": pl.Date,
            "expected_count": pl.Int64,
            "observed_count": pl.Int64,
            "coverage_ratio": pl.Float64,
        },
        "weights": {
            "time": pl.Date,
            "interval_end": pl.Date,
            "available_date": pl.Date,
            "asset_id": pl.String,
            "gross_weight": pl.Float64,
            "net_weight": pl.Float64,
        },
    }
    tables = {
        name: pl.DataFrame(rows, schema=schemas[name])
        for name, rows in (
            ("returns", return_rows),
            ("turnover", turnover_rows),
            ("value", value_rows),
            ("coverage", coverage_rows),
        )
    }
    tables["weights"] = (
        pl.concat(holdings) if holdings else pl.DataFrame(schema=schemas["weights"])
    )
    tables["target_weights"] = weights
    return tables, final_checkpoint
