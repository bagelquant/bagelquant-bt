"""Deterministic signal-to-weight portfolio policies."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from bagelquant_core import Panel

from .engine import backtest_weight_frame
from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME, validate_panel_frame
from .signal import ScheduledSignal


@dataclass(frozen=True, slots=True)
class PortfolioBuild:
    weights: Panel
    skipped: pl.DataFrame


@dataclass(frozen=True, slots=True)
class EqualWeightPolicy:
    top_n: int

    def build(self, signals: ScheduledSignal, **_: object) -> PortfolioBuild:
        selected = _top_n(signals, self.top_n)
        return PortfolioBuild(
            _weight_panel(_normalise(selected, "_unit"), signals),
            _empty_skipped(),
        )


@dataclass(frozen=True, slots=True)
class FloatMarketCapWeightPolicy:
    top_n: int
    market_cap_column: str = "float_market_cap"

    def build(
        self,
        signals: ScheduledSignal,
        *,
        market_caps: pl.DataFrame | None = None,
        **_: object,
    ) -> PortfolioBuild:
        if market_caps is None:
            raise InputValidationError("float market-cap policy requires market_caps")
        selected = _top_n(signals, self.top_n)
        caps = validate_panel_frame(
            market_caps, label="market_caps", value_columns=(self.market_cap_column,)
        )
        weighted = selected.join(caps, on=[TIME, ASSET_ID], how="left")
        missing = weighted.filter(pl.col(self.market_cap_column).is_null())
        if missing.height:
            dates = ", ".join(
                str(value) for value in missing.get_column(TIME).unique().sort()
            )
            raise InputValidationError(
                f"missing float market cap for selected signals at: {dates}"
            )
        invalid = weighted.filter(
            ~pl.col(self.market_cap_column).is_finite()
            | (pl.col(self.market_cap_column) <= 0)
        )
        if invalid.height:
            raise InputValidationError(
                "float market caps must be finite and positive"
            )
        return PortfolioBuild(
            _weight_panel(
                _normalise(weighted, self.market_cap_column), signals
            ),
            _empty_skipped(),
        )


@dataclass(frozen=True, slots=True)
class TargetVolatilityPolicy:
    base: EqualWeightPolicy | FloatMarketCapWeightPolicy
    target_annual_volatility: float = 0.15
    lookback_sessions: int = 60
    annualization: int = 240
    max_gross_exposure: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.target_annual_volatility <= 0
            or self.lookback_sessions <= 1
            or self.annualization <= 0
        ):
            raise ValueError("target volatility settings must be positive")
        if not 0 < self.max_gross_exposure <= 1.0:
            raise ValueError("max_gross_exposure must be in (0, 1]")

    def build(
        self,
        signals: ScheduledSignal,
        *,
        prices: pl.DataFrame | None = None,
        config=None,
        **kwargs: object,
    ) -> PortfolioBuild:
        if prices is None or config is None:
            raise InputValidationError(
                "target-volatility policy requires prices and config"
            )
        base_panel = self.base.build(signals, **kwargs).weights
        base = _weight_frame(base_panel)
        history = backtest_weight_frame(base, prices, config=config).returns
        dates = base.select(TIME).unique().sort(TIME)
        scales = []
        for value in dates.get_column(TIME):
            sample = history.filter(pl.col(TIME) < value).tail(self.lookback_sessions)
            if sample.height < self.lookback_sessions:
                scales.append(
                    {
                        TIME: value,
                        "scale": None,
                        "reason": "insufficient_volatility_history",
                    }
                )
                continue
            volatility = (
                float(sample.get_column("gross_return").std()) * self.annualization**0.5
            )
            scale = (
                self.max_gross_exposure
                if volatility == 0
                else min(
                    self.max_gross_exposure, self.target_annual_volatility / volatility
                )
            )
            scales.append({TIME: value, "scale": scale, "reason": None})
        scale_frame = pl.DataFrame(scales)
        weights = (
            base.join(scale_frame.select(TIME, "scale"), on=TIME, how="left")
            .drop_nulls("scale")
            .with_columns((pl.col("weight") * pl.col("scale")).alias("weight"))
            .select(TIME, ASSET_ID, "weight")
        )
        skipped = scale_frame.filter(pl.col("scale").is_null()).select(TIME, "reason")
        return PortfolioBuild(
            Panel.from_domain(
                weights.rename({"weight": "value"}),
                signals.signal.domain,
                name="weights",
            ),
            skipped,
        )


def _top_n(signals: ScheduledSignal, top_n: int) -> pl.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not isinstance(signals, ScheduledSignal):
        raise TypeError("portfolio policies require a ScheduledSignal")
    frame = validate_panel_frame(
        signals.signal.collect(dense=False).rename({"value": "signal"}),
        label="signals",
        value_columns=("signal",),
    )
    return (
        frame.sort([TIME, "signal"], descending=[False, True])
        .with_columns(pl.int_range(1, pl.len() + 1).over(TIME).alias("_rank"))
        .filter(pl.col("_rank") <= top_n)
        .with_columns(pl.lit(1.0).alias("_unit"))
    )


def _normalise(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return (
        frame.with_columns(
            (pl.col(column) / pl.col(column).sum().over(TIME)).alias("weight")
        )
        .select(TIME, ASSET_ID, "weight")
        .sort([TIME, ASSET_ID])
    )


def _empty_skipped() -> pl.DataFrame:
    return pl.DataFrame(schema={TIME: pl.Date, "reason": pl.String})


def _weight_panel(frame: pl.DataFrame, signals: ScheduledSignal) -> Panel:
    return Panel.from_domain(
        frame.rename({"weight": "value"}),
        signals.signal.domain,
        name="weights",
    )


def _weight_frame(weights: Panel) -> pl.DataFrame:
    return weights.collect(dense=False).drop_nulls("value").rename(
        {"value": "weight"}
    )
