"""Return-series utilities for long-form Polars panels."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import polars as pl

from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME


@dataclass(frozen=True, slots=True)
class PreparedPriceData:
    """Observed prices, daily valuation prices, returns, and price-gap evidence."""

    observed_prices: pl.DataFrame
    forward_returns: pl.DataFrame
    price_gaps: pl.DataFrame
    _valuation_prices: pl.DataFrame | None = None

    @property
    def valuation_prices(self) -> pl.DataFrame:
        """Materialize daily valuation prices only when a caller requests them."""

        cached = self._valuation_prices
        if cached is None:
            cached = _build_valuation_prices(self.observed_prices).select(
                TIME, ASSET_ID, "price"
            )
            object.__setattr__(self, "_valuation_prices", cached)
        return cached


@dataclass(frozen=True, slots=True)
class _ExpandedPortfolioWeights:
    """Daily target weights plus their sparse state-change events."""

    weights: pl.DataFrame
    target_events: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _PortfolioTargets:
    """Tradable snapshots and their non-redundant state changes."""

    snapshots: pl.DataFrame
    events: pl.DataFrame


def prepare_price_data(prices: pl.DataFrame) -> PreparedPriceData:
    """Build a shared daily valuation calendar without inventing tradable prices.

    The market calendar is the union of observed price dates.  Each asset is
    marked at its last observed price during a single-asset gap, so the gap has
    zero return and the cumulative move is recognized only on resumption.
    """

    return _prepare_price_data(prices, inputs_sorted=False)


def _prepare_price_data(
    prices: pl.DataFrame,
    *,
    inputs_sorted: bool,
) -> PreparedPriceData:
    """Internal variant that can reuse an already validated key order."""

    observed_prices = prices if inputs_sorted else prices.sort([TIME, ASSET_ID])
    sessions = observed_prices.get_column(TIME).unique(
        maintain_order=True
    ).to_frame().with_row_index("_session_index")
    asset_count = observed_prices.get_column(ASSET_ID).n_unique()
    if observed_prices.height == sessions.height * asset_count:
        forward_returns = (
            observed_prices.with_columns(
                (
                    pl.col("price").shift(-1).over(ASSET_ID) / pl.col("price") - 1.0
                ).alias("forward_return")
            )
            .drop_nulls("forward_return")
            .select(TIME, ASSET_ID, "forward_return")
        )
        price_gaps = _empty_price_gaps()
    else:
        forward_returns, price_gaps = _build_sparse_price_outputs(
            observed_prices,
            sessions,
        )
    return PreparedPriceData(
        observed_prices=observed_prices,
        forward_returns=forward_returns,
        price_gaps=price_gaps,
    )


def _build_sparse_price_outputs(
    observed_prices: pl.DataFrame,
    sessions: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Expand observed-price intervals directly into returns and gap evidence."""

    last_session = sessions.height - 1
    segments = (
        observed_prices.join(sessions, on=TIME, how="left")
        .sort([ASSET_ID, "_session_index"])
        .with_columns(
            pl.col("_session_index")
            .shift(-1)
            .over(ASSET_ID)
            .alias("_next_session"),
            pl.col("price").shift(-1).over(ASSET_ID).alias("_next_price"),
        )
    )
    zero_returns = (
        segments.select(
            ASSET_ID,
            pl.int_ranges(
                pl.col("_session_index").cast(pl.Int64),
                pl.when(pl.col("_next_session").is_not_null())
                .then(pl.col("_next_session").cast(pl.Int64) - 1)
                .otherwise(last_session),
            ).alias("_return_session"),
        )
        .explode("_return_session", empty_as_null=True)
        .drop_nulls("_return_session")
        .with_columns(pl.lit(0.0).alias("forward_return"))
    )
    movement_returns = segments.filter(
        pl.col("_next_session").is_not_null()
    ).select(
        ASSET_ID,
        (pl.col("_next_session") - 1).alias("_return_session"),
        (pl.col("_next_price") / pl.col("price") - 1.0).alias("forward_return"),
    )
    forward_returns = (
        pl.concat([zero_returns, movement_returns], how="vertical_relaxed")
        .join(
            sessions.select(
                pl.col("_session_index").alias("_return_session"),
                TIME,
            ),
            on="_return_session",
            how="left",
        )
        .select(TIME, ASSET_ID, "forward_return")
        .sort([TIME, ASSET_ID])
    )

    price_gaps = (
        segments.select(
            ASSET_ID,
            pl.col(TIME).alias("last_observed_time"),
            pl.col("_session_index").alias("_last_observed_session"),
            pl.int_ranges(
                pl.col("_session_index").cast(pl.Int64) + 1,
                pl.when(pl.col("_next_session").is_not_null())
                .then(pl.col("_next_session").cast(pl.Int64))
                .otherwise(last_session + 1),
            ).alias("_gap_session"),
        )
        .explode("_gap_session", empty_as_null=True)
        .drop_nulls("_gap_session")
        .join(
            sessions.select(
                pl.col("_session_index").alias("_gap_session"),
                TIME,
            ),
            on="_gap_session",
            how="left",
        )
        .select(
            TIME,
            ASSET_ID,
            "last_observed_time",
            (pl.col("_gap_session") - pl.col("_last_observed_session"))
            .cast(pl.Int64)
            .alias("missing_session_count"),
        )
        .sort([TIME, ASSET_ID])
    )
    return forward_returns, price_gaps


def _empty_price_gaps() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            TIME: pl.Date,
            ASSET_ID: pl.String,
            "last_observed_time": pl.Date,
            "missing_session_count": pl.Int64,
        }
    )


def _build_valuation_prices(observed_prices: pl.DataFrame) -> pl.DataFrame:
    sessions = observed_prices.select(TIME).unique().sort(TIME).with_row_index(
        "_session_index"
    )
    assets = observed_prices.select(ASSET_ID).unique()
    return (
        sessions.lazy()
        .join(assets.lazy(), how="cross")
        .join(observed_prices.lazy(), on=[TIME, ASSET_ID], how="left")
        .sort([ASSET_ID, TIME])
        .with_columns(
            pl.col("price").alias("_observed_price"),
            pl.when(pl.col("price").is_not_null())
            .then(pl.col("_session_index"))
            .otherwise(None)
            .forward_fill()
            .over(ASSET_ID)
            .alias("_last_observed_session"),
            pl.when(pl.col("price").is_not_null())
            .then(pl.col(TIME))
            .otherwise(None)
            .forward_fill()
            .over(ASSET_ID)
            .alias("_last_observed_time"),
            pl.col("price").forward_fill().over(ASSET_ID),
        )
        .collect(engine="streaming")
    )


def asset_close_to_close_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute daily close-to-close returns on the market session calendar."""

    return prepare_price_data(prices).forward_returns


def align_weights_to_forward_returns(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    price_data = prepare_price_data(prices)
    executable = _expand_portfolio_weights(
        weights,
        price_data.observed_prices,
        price_data.forward_returns,
    )
    return executable, price_data.forward_returns


def align_signal_to_forward_returns(
    signal: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    value_column: str,
    label: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep signal snapshots with exact price keys."""

    returns = asset_close_to_close_returns(prices)
    aligned = _filter_snapshots_to_price_keys(
        signal,
        prices,
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
    *,
    initial_target_weights: pl.DataFrame | None = None,
) -> pl.DataFrame:
    return _expand_portfolio_weight_data(
        weights,
        prices,
        forward_returns,
        initial_target_weights=initial_target_weights,
        include_target_events=False,
    ).weights


def _expand_portfolio_weight_data(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    initial_target_weights: pl.DataFrame | None = None,
    include_target_events: bool = True,
    execution_keys: pl.DataFrame | None = None,
) -> _ExpandedPortfolioWeights:
    if weights.is_empty():
        empty = weights.select(TIME, ASSET_ID, "weight")
        return _ExpandedPortfolioWeights(empty, empty)
    if prices.is_empty():
        raise InputValidationError("weights has no covered price range")

    first_execution_time = weights.get_column(TIME).min()
    resolved_execution_keys = (
        (
            forward_returns.select(TIME, ASSET_ID)
            .unique()
            .sort([ASSET_ID, TIME])
        )
        if execution_keys is None
        else execution_keys
    )
    resolved_execution_keys = (
        resolved_execution_keys
        .filter(pl.col(TIME) >= first_execution_time)
    )
    targets = _portfolio_targets(
        weights,
        prices,
        initial_target_weights=initial_target_weights,
        include_events=include_target_events,
    )
    tradable_targets = targets.snapshots
    target_events = targets.events
    initial = (
        pl.DataFrame(
            schema={ASSET_ID: pl.String, "_initial_weight": pl.Float64}
        )
        if initial_target_weights is None
        else initial_target_weights.select(
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("weight").cast(pl.Float64).alias("_initial_weight"),
        )
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        expanded = resolved_execution_keys.join_asof(
            tradable_targets.select(TIME, ASSET_ID, "weight"),
            on=TIME,
            by=ASSET_ID,
            strategy="backward",
        )
    if initial_target_weights is not None:
        expanded = (
            expanded.join(initial, on=ASSET_ID, how="left")
            .with_columns(
                pl.col("weight")
                .fill_null(pl.col("_initial_weight"))
                .alias("weight")
            )
            .drop("_initial_weight")
        )
    return _ExpandedPortfolioWeights(
        expanded
        .drop_nulls("weight")
        .select(TIME, ASSET_ID, "weight")
        .sort([TIME, ASSET_ID]),
        target_events,
    )


def _portfolio_targets(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    initial_target_weights: pl.DataFrame | None = None,
    include_events: bool = True,
) -> _PortfolioTargets:
    """Build the snapshot grid without expanding it across daily returns."""

    snapshot_times = (
        weights.select(TIME)
        .unique()
        .join(prices.select(TIME).unique(), on=TIME, how="inner")
    )
    snapshot_assets = prices.select(ASSET_ID).unique()
    target_snapshots = (
        snapshot_times.join(snapshot_assets, how="cross")
        .join(weights, on=[TIME, ASSET_ID], how="left")
        .with_columns(pl.col("weight").fill_null(0.0))
        .join(
            prices.select(TIME, ASSET_ID, pl.col("price").alias("_observed_price")),
            on=[TIME, ASSET_ID],
            how="left",
        )
        .sort([ASSET_ID, TIME])
    )
    tradable_targets = target_snapshots.filter(pl.col("_observed_price").is_not_null())
    initial = (
        pl.DataFrame(
            schema={ASSET_ID: pl.String, "_initial_weight": pl.Float64}
        )
        if initial_target_weights is None
        else initial_target_weights.select(
            pl.col(ASSET_ID).cast(pl.String),
            pl.col("weight").cast(pl.Float64).alias("_initial_weight"),
        )
    )
    target_events = (
        (
            tradable_targets.select(TIME, ASSET_ID, "weight")
            .join(initial, on=ASSET_ID, how="left")
            .with_columns(
                pl.col("weight")
                .shift(1)
                .over(ASSET_ID)
                .fill_null(pl.col("_initial_weight"))
                .fill_null(0.0)
                .alias("_previous_weight")
            )
            .filter(pl.col("weight") != pl.col("_previous_weight"))
            .select(TIME, ASSET_ID, "weight")
            .sort([TIME, ASSET_ID])
        )
        if include_events
        else weights.head(0).select(TIME, ASSET_ID, "weight")
    )
    return _PortfolioTargets(
        tradable_targets.select(TIME, ASSET_ID, "weight"),
        target_events,
    )


def unexecuted_weight_keys(weights: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Describe rebalance targets blocked by an absent executable price."""

    if weights.is_empty() or prices.is_empty():
        return _empty_unexecuted_weight_keys()
    snapshot_times = (
        weights.select(TIME)
        .unique()
        .join(prices.select(TIME).unique(), on=TIME, how="inner")
    )
    snapshot_assets = prices.select(ASSET_ID).unique()
    target_snapshots = (
        snapshot_times.join(snapshot_assets, how="cross")
        .join(weights, on=[TIME, ASSET_ID], how="left")
        .with_columns(pl.col("weight").fill_null(0.0).alias("target_weight"))
        .drop("weight")
        .join(
            prices.select(TIME, ASSET_ID, pl.col("price").alias("_observed_price")),
            on=[TIME, ASSET_ID],
            how="left",
        )
        .sort([ASSET_ID, TIME])
    )
    tradable_targets = target_snapshots.filter(pl.col("_observed_price").is_not_null())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        prior_targets = target_snapshots.join_asof(
            tradable_targets.select(TIME, ASSET_ID, "target_weight"),
            on=TIME,
            by=ASSET_ID,
            strategy="backward",
            suffix="_executed",
        )
    return (
        prior_targets.filter(
            pl.col("_observed_price").is_null()
            & (
                pl.col("target_weight")
                != pl.col("target_weight_executed").fill_null(0.0)
            )
        )
        .select(
            TIME,
            ASSET_ID,
            "target_weight",
            pl.col("target_weight_executed").fill_null(0.0).alias("retained_weight"),
        )
        .sort([TIME, ASSET_ID])
    )


def _empty_unexecuted_weight_keys() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            TIME: pl.Date,
            ASSET_ID: pl.String,
            "target_weight": pl.Float64,
            "retained_weight": pl.Float64,
        }
    )


def _filter_snapshots_to_price_keys(
    frame: pl.DataFrame,
    prices: pl.DataFrame,
    *,
    value_columns: tuple[str, ...],
    label: str,
) -> pl.DataFrame:
    if frame.is_empty():
        return frame.select(TIME, ASSET_ID, *value_columns)
    if prices.is_empty():
        raise InputValidationError(f"{label} has no covered price range")

    aligned = frame.join(
        prices.select(TIME, ASSET_ID),
        on=[TIME, ASSET_ID],
        how="inner",
    )
    return aligned.select(TIME, ASSET_ID, *value_columns).sort([TIME, ASSET_ID])


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
