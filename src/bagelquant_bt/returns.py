"""Return-series utilities for long-form Polars panels."""

from __future__ import annotations

from bisect import bisect_left

import polars as pl

from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME


def asset_close_to_close_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute close-to-close returns per asset."""

    return (
        prices.sort([ASSET_ID, TIME])
        .with_columns(
            (pl.col("price").shift(-1).over(ASSET_ID) / pl.col("price") - 1.0).alias(
                "forward_return"
            )
        )
        .filter(pl.col("forward_return").is_not_null())
        .select(TIME, ASSET_ID, "forward_return")
        .sort([TIME, ASSET_ID])
    )


def align_weights_to_forward_returns(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    returns = asset_close_to_close_returns(prices)
    executable = _expand_portfolio_weights(weights, prices, returns)
    return executable, returns


def align_signal_to_forward_returns(
    signal: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    value_column: str,
    label: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Map sparse signal snapshots to the next tradable price date."""

    returns = asset_close_to_close_returns(prices)
    aligned = _align_snapshots_to_tradable_times(
        signal,
        prices,
        returns,
        value_columns=(value_column,),
        label=label,
    )
    return aligned, returns


def portfolio_returns(
    weights: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    return (
        weights.join(forward_returns, on=[TIME, ASSET_ID], how="inner")
        .with_columns(
            (
                pl.col("weight").fill_null(0.0)
                * pl.col("forward_return").fill_null(0.0)
            ).alias("weighted_return")
        )
        .group_by(TIME)
        .agg(pl.col("weighted_return").sum().alias("gross_return"))
        .sort(TIME)
    )


def _expand_portfolio_weights(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
) -> pl.DataFrame:
    aligned = _align_snapshots_to_tradable_times(
        weights,
        prices,
        forward_returns,
        value_columns=("weight",),
        label="weights",
    )
    if aligned.is_empty():
        return aligned

    tradable_times = _sorted_unique(forward_returns[TIME].to_list())
    snapshots = {
        _partition_key(time): {
            str(row[ASSET_ID]): 0.0 if row["weight"] is None else float(row["weight"])
            for row in group.iter_rows(named=True)
        }
        for time, group in aligned.partition_by(TIME, as_dict=True).items()
    }
    first_execution_time = min(snapshots)
    current: dict[str, float] | None = None
    rows: list[dict[str, object]] = []
    for time in tradable_times:
        if time < first_execution_time:
            continue
        if time in snapshots:
            next_target = snapshots[time]
            previous_assets = set(current or {})
            current = {
                asset: next_target.get(asset, 0.0)
                for asset in previous_assets | set(next_target)
            }
        if current is None:
            continue
        for asset, weight in current.items():
            rows.append({TIME: time, ASSET_ID: asset, "weight": weight})

    return pl.DataFrame(
        rows,
        schema={TIME: pl.Date, ASSET_ID: pl.String, "weight": pl.Float64},
    ).sort([TIME, ASSET_ID])


def _align_snapshots_to_tradable_times(
    frame: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    value_columns: tuple[str, ...],
    label: str,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame.select(TIME, ASSET_ID, *value_columns)
    if forward_returns.is_empty():
        raise InputValidationError(f"{label} has no covered price range")

    tradable_times = _sorted_unique(forward_returns[TIME].to_list())
    price_times = _sorted_unique(prices[TIME].to_list())
    first_price_time = price_times[0]
    last_price_time = price_times[-1]
    return_keys = {
        (row[TIME], str(row[ASSET_ID]))
        for row in forward_returns.select(TIME, ASSET_ID).iter_rows(named=True)
    }
    price_assets = {asset for _, asset in return_keys}
    snapshots: dict[object, pl.DataFrame] = {}

    for source_time, group in frame.partition_by(TIME, as_dict=True).items():
        source_time = _partition_key(source_time)
        if source_time < first_price_time or source_time > last_price_time:
            raise InputValidationError(
                f"{label} time is outside covered price range: {source_time}"
            )
        execution_time = _next_tradable_time(source_time, tradable_times)
        if execution_time is None:
            continue

        assets = [str(asset) for asset in group[ASSET_ID].to_list()]
        missing_assets = sorted(set(assets) - price_assets)
        if missing_assets:
            raise InputValidationError(
                f"{label} contains assets missing from prices: {missing_assets}"
            )
        missing_at_execution = sorted(
            asset
            for asset in set(assets)
            if (execution_time, asset) not in return_keys
        )
        if missing_at_execution:
            raise InputValidationError(
                f"{label} assets are not tradable at {execution_time}: "
                f"{missing_at_execution}"
            )

        snapshots[execution_time] = group.with_columns(
            pl.lit(execution_time).alias(TIME)
        )

    aligned = pl.concat(snapshots.values()) if snapshots else frame.clear()
    return aligned.select(TIME, ASSET_ID, *value_columns).sort([TIME, ASSET_ID])


def _next_tradable_time(
    source_time: object,
    tradable_times: list[object],
) -> object | None:
    index = bisect_left(tradable_times, source_time)
    if index >= len(tradable_times):
        return None
    return tradable_times[index]


def _sorted_unique(values: list[object]) -> list[object]:
    return sorted(set(values))


def _partition_key(key: object) -> object:
    if isinstance(key, tuple):
        return key[0]
    if isinstance(key, list):
        return key[0]
    return key


def cumulative_returns(returns: pl.DataFrame, column: str) -> pl.DataFrame:
    return returns.select(
        TIME,
        ((1.0 + pl.col(column).fill_null(0.0)).cum_prod() - 1.0).alias(
            f"{column}_cumulative"
        ),
    )


def value_path(
    returns: pl.DataFrame,
    column: str,
    *,
    initial_capital: float,
    output_column: str,
) -> pl.DataFrame:
    return returns.select(
        TIME,
        (initial_capital * (1.0 + pl.col(column).fill_null(0.0)).cum_prod()).alias(
            output_column
        ),
    )


def drawdown(returns: pl.DataFrame, column: str) -> pl.DataFrame:
    wealth = (1.0 + pl.col(column).fill_null(0.0)).cum_prod()
    return returns.select(TIME, (wealth / wealth.cum_max() - 1.0).alias("drawdown"))
