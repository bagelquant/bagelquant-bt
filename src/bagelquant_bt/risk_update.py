"""Metadata planning and bounded maturity transitions for daily risk fits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import cast

import polars as pl

from .inputs import validate_prices
from .risk import fit_risk_factor_returns, risk_interval_returns

RISK_UPDATE_POLICY = "daily_risk_maturity.v2"


@dataclass(frozen=True, slots=True)
class RiskFactorUpdatePlan:
    """Only decisions whose next-open interval becomes newly/again available."""

    calendar: tuple[date, ...]
    start: date
    through: date
    update_start: date
    completed_through: date | None
    schedule: pl.DataFrame
    mode: str

    def __post_init__(self):
        if self.mode not in {"initialize", "append", "repair", "up_to_date"}:
            raise ValueError("unknown risk update mode")
        if self.schedule.schema != {
            "decision_date": pl.Date,
            "time": pl.Date,
            "available_date": pl.Date,
        }:
            raise TypeError(
                "risk schedule requires typed decision/start/availability dates"
            )
        if (
            any(self.schedule[name].null_count() for name in self.schedule.columns)
            or self.schedule["available_date"].n_unique() != self.schedule.height
        ):
            raise ValueError("risk maturity schedule must be nonnull and unique")
        triples = set(
            zip(self.calendar, self.calendar[1:], self.calendar[2:], strict=False)
        )
        if any(
            (row["decision_date"], row["time"], row["available_date"]) not in triples
            or not self.update_start <= row["available_date"] <= self.through
            for row in self.schedule.iter_rows(named=True)
        ):
            raise ValueError("risk schedule differs from admitted session maturity")

    @property
    def complete(self):
        return self.mode == "up_to_date"

    @property
    def exposure_start(self) -> date | None:
        return (
            cast(date, self.schedule["decision_date"].min())
            if self.schedule.height
            else None
        )

    @property
    def price_start(self) -> date | None:
        return cast(date, self.schedule["time"].min()) if self.schedule.height else None

    @property
    def carry_boundary(self):
        admitted = [day for day in self.calendar if day <= self.through]
        return (
            admitted[-2]
            if len(admitted) >= 2
            else admitted[-1]
            if admitted
            else self.through
        )

    def windows(self, *, sessions: int = 32):
        """Yield bounded independent fit batches; no monthly model or alias exists."""
        if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 1:
            raise ValueError("risk batch sessions must be positive")
        if self.complete:
            return
        if not self.schedule.height:
            yield self
            return
        previous = self.completed_through
        for offset in range(0, self.schedule.height, sessions):
            schedule = self.schedule.slice(offset, sessions)
            last = (
                cast(date, schedule["available_date"].max())
                if offset + sessions < self.schedule.height
                else self.through
            )
            yield RiskFactorUpdatePlan(
                self.calendar,
                self.start,
                last,
                cast(date, schedule["available_date"].min()),
                previous,
                schedule,
                self.mode,
            )
            previous = last


def plan_risk_factor_update(
    calendar: Sequence[date],
    *,
    start: date,
    through: date,
    completed_through: date | None = None,
    affected_available_date: date | None = None,
) -> RiskFactorUpdatePlan:
    """Plan from owner-verified completion evidence without prices or exposures.

    A changed completed prefix requires its first affected availability date;
    callers must not supply stale completion evidence as a currentness proof.
    The final two decisions are pending until their following-open interval
    has matured. Gaps never compress the admitted session calendar.
    """
    sessions = tuple(calendar)
    if tuple(sorted(set(sessions))) != sessions or any(
        type(day) is not date for day in sessions
    ):
        raise ValueError("risk calendar must be sorted unique dates")
    if start > through or (
        affected_available_date is not None and affected_available_date > through
    ):
        raise ValueError("risk update boundaries are reversed")
    first = (
        start
        if completed_through is None
        else max(start, completed_through + timedelta(days=1))
    )
    if affected_available_date is not None:
        first = min(first, max(start, affected_available_date))
    mode = (
        "initialize"
        if completed_through is None
        else "repair"
        if affected_available_date is not None
        else "append"
    )
    if first > through:
        mode = "up_to_date"
    rows = [
        {
            "decision_date": sessions[index],
            "time": sessions[index + 1],
            "available_date": sessions[index + 2],
        }
        for index in range(max(0, len(sessions) - 2))
        if first <= sessions[index + 2] <= through
    ]
    schedule = pl.DataFrame(
        rows,
        schema={"decision_date": pl.Date, "time": pl.Date, "available_date": pl.Date},
    )
    return RiskFactorUpdatePlan(
        sessions, start, through, first, completed_through, schedule, mode
    )


@dataclass(frozen=True, slots=True)
class RiskFactorResult:
    """BT-owned numerical primitives and compact continuation state.

    Publish ``factor_returns``, ``diagnostics`` and ``price_carry`` with
    BTStore.publish_shared; append old primitives through its immutable
    reference composition/publish_continuation APIs. The compact state belongs
    in receipt metadata. Core exposure identities and source-prefix proofs are
    opaque receipt bindings provided by the host, never resolved by BT.
    """

    factor_returns: pl.DataFrame
    diagnostics: pl.DataFrame
    price_carry: pl.DataFrame
    state: dict
    fit_count: int
    industry_activity: pl.DataFrame = field(default_factory=pl.DataFrame)


def evaluate_risk_factor_returns(
    exposures: pl.DataFrame,
    prices: pl.DataFrame,
    plan: RiskFactorUpdatePlan,
    *,
    styles: Sequence[str],
    industry_controls: bool = True,
    weighting: str = "sqrt_cap",
    minimum_assets: int = 200,
    observations_per_parameter: int = 5,
    population: pl.DataFrame | None = None,
    require_continuity: bool = False,
    valid_start: date | None = None,
    check_canceled: Callable[[], None] = lambda: None,
) -> RiskFactorResult:
    """Fit exactly admitted maturity dates, retaining original price carry dates.

    Prices include only the admitted bounded delta plus one preceding actual
    observation per asset. ``population`` optionally supplies the exact
    diagnostic time/asset sample before complete-case exclusion, preserving a
    host's declared historical population without changing valid fit rows.
    """
    if plan.complete:
        raise ValueError("a completed risk update needs no numerical execution")
    if valid_start is not None and type(valid_start) is not date:
        raise ValueError("valid_start must be a Date")
    if len(set(styles)) != len(styles) or any(
        not isinstance(style, str) or not style for style in styles
    ):
        raise ValueError("risk styles must be unique nonempty names")
    if not styles and not industry_controls:
        raise ValueError("risk fitting requires styles or industry controls")
    if weighting not in {"sqrt_cap", "equal"}:
        raise ValueError("risk weighting must be sqrt_cap or equal")
    required = {"time", "asset_id", *styles}
    numeric = ["forward_return", *styles]
    if weighting == "sqrt_cap":
        required.add("market_cap")
        numeric.append("market_cap")
    if industry_controls:
        required.add("industry")
    if not required <= set(exposures.columns):
        raise ValueError(
            f"missing risk exposures: {sorted(required - set(exposures.columns))}"
        )
    if exposures.select("time", "asset_id").unique().height != exposures.height:
        raise ValueError("risk exposures require unique decision/asset coordinates")
    prices = validate_prices(prices)
    if prices.filter(pl.col("time") > plan.through).height:
        raise ValueError("risk prices exceed the admitted information boundary")
    if population is not None:
        population = population.select("time", "asset_id")
        if population.unique().height != population.height:
            raise ValueError(
                "diagnostic population requires unique decision/asset coordinates"
            )
        selected = exposures.filter(
            pl.col("time").is_in(plan.schedule["decision_date"].implode())
        )
        if not selected.join(
            population, on=["time", "asset_id"], how="anti"
        ).is_empty():
            raise ValueError("diagnostic population cannot omit admitted exposure rows")
    return_calendar = pl.DataFrame(
        {
            "time": [
                day
                for day in plan.calendar
                if plan.price_start is not None
                and plan.price_start <= day <= plan.through
            ]
        },
        schema={"time": pl.Date},
    )
    forward = risk_interval_returns(prices, return_calendar)
    exposure_groups = {
        part["time"][0]: part
        for part in exposures.partition_by("time", maintain_order=True)
    }
    return_groups = {
        part["time"][0]: part
        for part in forward.partition_by("time", maintain_order=True)
    }
    population_groups = (
        {}
        if population is None
        else {
            part["time"][0]: part
            for part in population.partition_by("time", maintain_order=True)
        }
    )
    factors, diagnostics, activity = [], [], []
    for row in plan.schedule.iter_rows(named=True):
        check_canceled()
        cross = exposure_groups.get(row["decision_date"], exposures.head(0))
        if population is not None:
            cross = population_groups.get(
                row["decision_date"], population.head(0)
            ).join(cross, on=["time", "asset_id"], how="left")
        cross = cross.join(
            return_groups.get(row["time"], forward.head(0)).select(
                "asset_id", "forward_return"
            ),
            on="asset_id",
            how="left",
        )
        fit = fit_risk_factor_returns(
            cross,
            styles=styles,
            industry_controls=industry_controls,
            weighting=weighting,
            minimum_assets=minimum_assets,
            observations_per_parameter=observations_per_parameter,
        )
        if industry_controls:
            known = cross.filter(
                pl.col("industry").is_not_null()
                & (pl.col("industry").cast(pl.String).str.len_chars() > 0)
            )
            for group in known.partition_by("industry"):
                valid = pl.all_horizontal(
                    [pl.col(name).is_finite().fill_null(False) for name in numeric]
                )
                if weighting == "sqrt_cap":
                    valid &= pl.col("market_cap") > 0
                complete = group.filter(valid)
                activity.append(
                    {
                        **row,
                        "factor": "industry:" + str(group["industry"][0]),
                        "active": True,
                        "eligible_count": group.height,
                        "sample_count": complete.height,
                    }
                )
            # A dated anchor certifies the complete activity inventory even
            # when no industry has members. Absence is never inferred from fits.
            activity.append(
                {
                    **row,
                    "factor": "market",
                    "active": True,
                    "eligible_count": cross.height,
                    "sample_count": fit.sample_count,
                }
            )
        factors.append(
            fit.factor_returns.with_columns(
                pl.lit(row["time"]).alias("time"),
                pl.lit(row["decision_date"]).alias("decision_date"),
                pl.lit(row["available_date"]).alias("available_date"),
                pl.lit(True).alias("active"),
                pl.lit(fit.reason).cast(pl.String).alias("reason"),
            )
        )
        diagnostics.append(
            {
                "time": row["time"],
                "decision_date": row["decision_date"],
                "available_date": row["available_date"],
                **{
                    key: value
                    for key, value in asdict(fit).items()
                    if key != "factor_returns"
                },
            }
        )
    returns = (
        pl.concat(factors)
        if factors
        else pl.DataFrame(
            schema={
                "factor": pl.String,
                "family": pl.String,
                "factor_return": pl.Float64,
                "time": pl.Date,
                "decision_date": pl.Date,
                "available_date": pl.Date,
                "active": pl.Boolean,
                "reason": pl.String,
            }
        )
    )
    diagnostic_frame = pl.DataFrame(
        diagnostics,
        schema={
            "time": pl.Date,
            "decision_date": pl.Date,
            "available_date": pl.Date,
            "sample_count": pl.Int64,
            "eligible_count": pl.Int64,
            "excluded_count": pl.Int64,
            "cap_coverage": pl.Float64,
            "rank": pl.Int64,
            "parameter_count": pl.Int64,
            "condition_number": pl.Float64,
            "reason": pl.String,
        },
    )
    for diagnostic in diagnostic_frame.sort("time").iter_rows(named=True):
        if (
            require_continuity
            and valid_start is not None
            and diagnostic["reason"] is None
            and diagnostic["time"] < valid_start
        ):
            raise ValueError(
                "RISK_START_REPAIR_REQUIRED: explicitly audit "
                "the earlier supported prefix"
            )
        if valid_start is None and diagnostic["reason"] is None:
            valid_start = diagnostic["time"]
        if (
            require_continuity
            and valid_start is not None
            and diagnostic["time"] >= valid_start
            and diagnostic["reason"] is not None
        ):
            raise ValueError(
                f"RISK_CONTINUITY_REQUIRED: {diagnostic['time']}: "
                f"{diagnostic['reason']}; start={valid_start}"
            )
    carry = (
        prices.filter(pl.col("time") <= plan.carry_boundary)
        .sort("asset_id", "time")
        .unique("asset_id", keep="last", maintain_order=True)
        .sort("asset_id")
    )
    state = {
        "policy": RISK_UPDATE_POLICY,
        "through": plan.through.isoformat(),
        "carry_boundary": plan.carry_boundary.isoformat(),
        "styles": list(styles),
        "industry_controls": industry_controls,
        "weighting": weighting,
        "valid_start": None if valid_start is None else str(valid_start),
        "fit_coverage": {
            "times": [str(day) for day in diagnostic_frame["time"]],
            "first_valid": (
                None
                if diagnostic_frame.filter(pl.col("reason").is_null()).is_empty()
                else str(
                    diagnostic_frame.filter(pl.col("reason").is_null())["time"].min()
                )
            ),
            "unavailable": [
                {"time": str(row["time"]), "reason": row["reason"]}
                for row in diagnostic_frame.filter(
                    pl.col("reason").is_not_null()
                ).iter_rows(named=True)
            ],
        },
        "minimum_assets": minimum_assets,
        "observations_per_parameter": observations_per_parameter,
        "pending_decisions": [
            day.isoformat() for day in plan.calendar if day <= plan.through
        ][-2:],
    }
    check_canceled()
    return RiskFactorResult(
        returns.sort("time", "factor"),
        diagnostic_frame.sort("time"),
        carry,
        state,
        len(diagnostics),
        pl.DataFrame(
            activity,
            schema={
                "decision_date": pl.Date,
                "time": pl.Date,
                "available_date": pl.Date,
                "factor": pl.String,
                "active": pl.Boolean,
                "eligible_count": pl.Int64,
                "sample_count": pl.Int64,
            },
        ).sort("time", "factor"),
    )


__all__ = [
    "RISK_UPDATE_POLICY",
    "RiskFactorResult",
    "RiskFactorUpdatePlan",
    "evaluate_risk_factor_returns",
    "plan_risk_factor_update",
]
