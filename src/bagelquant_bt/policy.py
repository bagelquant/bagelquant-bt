"""Alpha evaluation-date processing and execution-date scheduling."""

from __future__ import annotations

import calendar as month_calendar
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Literal

import polars as pl
from bagelquant_core import Domain, Panel, PredictionPanel

from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME


class EvaluationFrequency(StrEnum):
    DAILY = "daily"
    MONTHLY = "monthly"


class EvaluationAnchor(StrEnum):
    EVERY_TRADING_DAY = "every_trading_day"
    LAST_TRADING_DAY = "last_trading_day"
    ON_OR_AFTER_CALENDAR_DAY = "on_or_after_calendar_day"
    FIRST_WEEKDAY = "first_weekday"
    LAST_WEEKDAY = "last_weekday"


class AlphaStandardization(StrEnum):
    NONE = "none"
    Z_SCORE = "z_score"
    PERCENTILE_RANK = "percentile_rank"


class HolidayAdjustment(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    NEXT_OPEN_SESSION = "next_open_session"
    PREVIOUS_OPEN_SESSION = "previous_open_session"


class MissingSnapshotAction(StrEnum):
    SKIP = "skip"
    PREVIOUS_IN_PERIOD = "previous_in_period"


@dataclass(frozen=True, slots=True)
class AlphaPolicyResult:
    """Evaluation schedule and one processed Panel for every AlphaValue."""

    schedule: pl.DataFrame
    alpha_values: Mapping[str, Panel]


@dataclass(frozen=True, slots=True)
class ScheduledPrediction:
    """Prediction values mapped to executable market sessions."""

    schedule: pl.DataFrame
    prediction: PredictionPanel

    @property
    def identity(self) -> str:
        payload = self.schedule.sort("requested_rebalance_date").write_json()
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


@dataclass(frozen=True, slots=True)
class ScheduledWeights:
    """Target weights mapped from evaluation dates to execution dates."""

    schedule: pl.DataFrame
    weights: Panel


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Map evaluation-date values to a later executable market session."""

    id: str
    lag_sessions: int = 1

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("execution policy requires an id")
        if self.lag_sessions <= 0:
            raise ValueError("execution lag must be positive")

    def resolve(
        self,
        rebalance_dates: pl.DataFrame,
        calendar: pl.DataFrame,
    ) -> pl.DataFrame:
        """Attach an execution date counted from each evaluation date."""

        if "rebalance_date" not in rebalance_dates.columns:
            raise InputValidationError(
                "rebalance schedule is missing required column: ['rebalance_date']"
            )
        sessions = _open_sessions(calendar)
        positions = {session: index for index, session in enumerate(sessions)}
        rows: list[dict[str, object]] = []
        for row in rebalance_dates.iter_rows(named=True):
            rebalance_date = row["rebalance_date"]
            position = positions.get(rebalance_date)
            execution_position = (
                None if position is None else position + self.lag_sessions
            )
            output = dict(row)
            output["execution_policy_id"] = self.id
            output["execution_date"] = (
                sessions[execution_position]
                if execution_position is not None
                and execution_position < len(sessions)
                else None
            )
            rows.append(output)
        return pl.DataFrame(rows) if rows else rebalance_dates.with_columns(
            pl.lit(self.id, dtype=pl.String).alias("execution_policy_id"),
            pl.lit(None, dtype=pl.Date).alias("execution_date"),
        )

    def schedule_prediction(
        self,
        prediction: PredictionPanel,
        calendar: pl.DataFrame,
    ) -> ScheduledPrediction:
        """Move a PredictionPanel to execution dates while retaining lineage."""

        if not isinstance(prediction, PredictionPanel):
            raise TypeError("ExecutionPolicy requires a PredictionPanel")
        values = prediction.collect(dense=False).drop_nulls("value")
        requested = _schedule_from_panel(values)
        schedule = self.resolve(requested, calendar).with_columns(
            pl.col("rebalance_date").alias("alpha_date"),
            pl.when(pl.col("execution_date").is_not_null())
            .then(pl.lit("exact"))
            .otherwise(pl.lit("skipped"))
            .alias("selection_status"),
            pl.when(pl.col("execution_date").is_null())
            .then(pl.lit("missing_execution_session"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("skip_reason"),
        )
        executable = schedule.drop_nulls("execution_date")
        mapped = (
            values.join(
                executable.select(
                    pl.col("rebalance_date").alias(TIME), "execution_date"
                ),
                on=TIME,
                how="inner",
            )
            .with_columns(
                pl.col(TIME).alias("source_time"),
                pl.col("execution_date").alias(TIME),
            )
            .select(TIME, ASSET_ID, "value")
            .sort([TIME, ASSET_ID])
        )
        return ScheduledPrediction(
            schedule=schedule,
            prediction=PredictionPanel.from_domain(
                mapped,
                _mapped_domain(executable, prediction),
                name=prediction.name,
                metadata={
                    **prediction.metadata,
                    "execution_policy": self.id,
                },
            ),
        )

    def schedule_weights(
        self,
        weights: Panel,
        calendar: pl.DataFrame,
    ) -> ScheduledWeights:
        """Move ordinary target weights to execution dates."""

        if not isinstance(weights, Panel) or isinstance(weights, PredictionPanel):
            raise TypeError("ExecutionPolicy requires an ordinary weights Panel")
        values = weights.collect(dense=False).drop_nulls("value")
        requested = _schedule_from_panel(values)
        schedule = self.resolve(requested, calendar)
        executable = schedule.drop_nulls("execution_date")
        mapped = (
            values.join(
                executable.select(
                    pl.col("rebalance_date").alias(TIME), "execution_date"
                ),
                on=TIME,
                how="inner",
            )
            .with_columns(pl.col("execution_date").alias(TIME))
            .select(TIME, ASSET_ID, "value")
            .sort([TIME, ASSET_ID])
        )
        return ScheduledWeights(
            schedule=schedule,
            weights=Panel.from_domain(
                mapped,
                _mapped_domain(executable, weights),
                name=weights.name,
                metadata={**weights.metadata, "execution_policy": self.id},
            ),
        )


@dataclass(frozen=True, slots=True)
class AlphaPolicy:
    """Align Alpha snapshots to evaluation dates, then standardize each Alpha."""

    id: str
    frequency: Literal["daily", "monthly"]
    anchor: EvaluationAnchor
    missing_snapshot: MissingSnapshotAction
    standardization: AlphaStandardization = AlphaStandardization.NONE
    calendar_day: int | None = None
    weekday: int | None = None
    observation_offset_sessions: int = 0
    holiday_adjustment: HolidayAdjustment = HolidayAdjustment.NOT_APPLICABLE

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("alpha policy requires an id")
        object.__setattr__(
            self, "standardization", AlphaStandardization(self.standardization)
        )
        if self.observation_offset_sessions < 0:
            raise ValueError("observation offset cannot be negative")
        if self.anchor == EvaluationAnchor.ON_OR_AFTER_CALENDAR_DAY:
            if self.calendar_day is None or not 1 <= self.calendar_day <= 28:
                raise ValueError("calendar-day anchor requires a day from 1 through 28")
        elif self.calendar_day is not None:
            raise ValueError("calendar_day is valid only for on-or-after anchor")
        if self.anchor in {
            EvaluationAnchor.FIRST_WEEKDAY,
            EvaluationAnchor.LAST_WEEKDAY,
        }:
            if self.weekday is None or not 0 <= self.weekday <= 6:
                raise ValueError("weekday anchor requires weekday from 0 through 6")
        elif self.weekday is not None:
            raise ValueError("weekday is valid only for weekday anchors")

    def schedule(
        self,
        calendar: pl.DataFrame,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        """Resolve requested evaluation sessions from an exchange calendar."""

        sessions = _open_sessions(calendar)
        positions = {session: index for index, session in enumerate(sessions)}
        rows: list[dict[str, object]] = []
        for anchored in _observations(sessions, self):
            alpha_position = positions[anchored] - self.observation_offset_sessions
            if alpha_position < 0:
                continue
            requested = sessions[alpha_position]
            if (start is not None and requested < start) or (
                end is not None and requested > end
            ):
                continue
            rows.append(
                {
                    "policy_id": self.id,
                    "period": (
                        requested.isoformat()
                        if self.frequency == "daily"
                        else requested.strftime("%Y-%m")
                    ),
                    "requested_rebalance_date": requested,
                    "rebalance_date": requested,
                }
            )
        return pl.DataFrame(
            rows,
            schema={
                "policy_id": pl.String,
                "period": pl.String,
                "requested_rebalance_date": pl.Date,
                "rebalance_date": pl.Date,
            },
        ).sort("rebalance_date")

    def apply(
        self,
        alpha_values: Mapping[str, Panel],
        calendar: pl.DataFrame,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> AlphaPolicyResult:
        """Apply evaluation-date alignment before cross-sectional processing."""

        if not alpha_values:
            raise ValueError("AlphaPolicy requires at least one AlphaValue Panel")
        if any(
            not isinstance(value, Panel) or isinstance(value, PredictionPanel)
            for value in alpha_values.values()
        ):
            raise TypeError("alpha_values must contain ordinary Panel values")
        domains = [value.domain for value in alpha_values.values()]
        if any(not domains[0].equivalent_to(domain) for domain in domains[1:]):
            raise ValueError("AlphaValue Panels must use equivalent Domains")
        schedule = self.schedule(calendar, start=start, end=end)
        if schedule.is_empty():
            raise InputValidationError("alpha policy resolves no evaluation dates")
        processed = {
            name: self._process_panel(value, schedule)
            for name, value in alpha_values.items()
        }
        return AlphaPolicyResult(schedule=schedule, alpha_values=processed)

    def _process_panel(self, alpha: Panel, schedule: pl.DataFrame) -> Panel:
        aligned = _align_alpha(alpha, schedule, self)
        value = pl.col("value").fill_nan(None)
        finite = pl.when(value.is_finite()).then(value).otherwise(None)
        if self.standardization == AlphaStandardization.NONE:
            expression = finite
        elif self.standardization == AlphaStandardization.Z_SCORE:
            deviation = finite.std(ddof=1).over(TIME)
            expression = pl.when(deviation.is_not_null() & (deviation > 0)).then(
                (finite - finite.mean().over(TIME)) / deviation
            )
        else:
            count = finite.count().over(TIME)
            expression = pl.when(count > 0).then(
                finite.rank("average").over(TIME) / count
            )
        frame = aligned.with_columns(expression.alias("value")).drop_nulls("value")
        domain = Domain(
            calendar=schedule.get_column("rebalance_date"),
            universe=alpha.domain.asset_ids,
        )
        return Panel.from_domain(
            frame,
            domain,
            name=alpha.name,
            metadata={
                **alpha.metadata,
                "alpha_policy": self.id,
                "standardization": self.standardization.value,
            },
        )


_CANONICAL_EXECUTION_POLICIES = {
    "next_open": ExecutionPolicy("next_open", lag_sessions=1),
}

_CANONICAL_ALPHA_POLICIES = {
    "daily": AlphaPolicy(
        "daily",
        "daily",
        EvaluationAnchor.EVERY_TRADING_DAY,
        MissingSnapshotAction.SKIP,
    ),
    "month_end": AlphaPolicy(
        "month_end",
        "monthly",
        EvaluationAnchor.LAST_TRADING_DAY,
        MissingSnapshotAction.PREVIOUS_IN_PERIOD,
    ),
    "monthly_mid": AlphaPolicy(
        "monthly_mid",
        "monthly",
        EvaluationAnchor.ON_OR_AFTER_CALENDAR_DAY,
        MissingSnapshotAction.SKIP,
        calendar_day=15,
        holiday_adjustment=HolidayAdjustment.NEXT_OPEN_SESSION,
    ),
    "monthly_first_monday": AlphaPolicy(
        "monthly_first_monday",
        "monthly",
        EvaluationAnchor.FIRST_WEEKDAY,
        MissingSnapshotAction.SKIP,
        weekday=0,
        holiday_adjustment=HolidayAdjustment.NEXT_OPEN_SESSION,
    ),
    "monthly_last_friday": AlphaPolicy(
        "monthly_last_friday",
        "monthly",
        EvaluationAnchor.LAST_WEEKDAY,
        MissingSnapshotAction.SKIP,
        weekday=4,
        holiday_adjustment=HolidayAdjustment.PREVIOUS_OPEN_SESSION,
    ),
}


def execution_policies() -> tuple[ExecutionPolicy, ...]:
    return tuple(_CANONICAL_EXECUTION_POLICIES.values())


def resolve_execution_policy(policy_id: str) -> ExecutionPolicy:
    try:
        return _CANONICAL_EXECUTION_POLICIES[policy_id]
    except KeyError as error:
        raise KeyError(f"unknown execution policy: {policy_id}") from error


def alpha_policies() -> tuple[AlphaPolicy, ...]:
    return tuple(_CANONICAL_ALPHA_POLICIES.values())


def resolve_alpha_policy(
    policy_id: str,
    *,
    standardization: AlphaStandardization | str | None = None,
) -> AlphaPolicy:
    try:
        policy = _CANONICAL_ALPHA_POLICIES[policy_id]
    except KeyError as error:
        raise KeyError(f"unknown alpha policy: {policy_id}") from error
    if standardization is None:
        return policy
    return AlphaPolicy(
        id=policy.id,
        frequency=policy.frequency,
        anchor=policy.anchor,
        missing_snapshot=policy.missing_snapshot,
        standardization=AlphaStandardization(standardization),
        calendar_day=policy.calendar_day,
        weekday=policy.weekday,
        observation_offset_sessions=policy.observation_offset_sessions,
        holiday_adjustment=policy.holiday_adjustment,
    )


def _align_alpha(
    alpha: Panel,
    schedule: pl.DataFrame,
    policy: AlphaPolicy,
) -> pl.DataFrame:
    values = alpha.collect(dense=False).drop_nulls("value")
    available = values.select(TIME).unique().sort(TIME).get_column(TIME)
    frames: list[pl.DataFrame] = []
    for row in schedule.iter_rows(named=True):
        rebalance = row["rebalance_date"]
        candidates = available.filter(available <= rebalance)
        if policy.frequency == "monthly":
            candidates = candidates.filter(
                candidates.dt.strftime("%Y-%m") == row["period"]
            )
        selected = (
            rebalance
            if rebalance in candidates
            else candidates.max()
            if policy.missing_snapshot == MissingSnapshotAction.PREVIOUS_IN_PERIOD
            and not candidates.is_empty()
            else None
        )
        if selected is not None:
            frames.append(
                values.filter(pl.col(TIME) == selected).with_columns(
                    pl.lit(rebalance, dtype=pl.Date).alias(TIME)
                )
            )
    if not frames:
        return pl.DataFrame(
            schema={TIME: pl.Date, ASSET_ID: pl.String, "value": pl.Float64}
        )
    return pl.concat(frames).sort([TIME, ASSET_ID])


def _schedule_from_panel(values: pl.DataFrame) -> pl.DataFrame:
    dates = values.select(TIME).unique().sort(TIME).get_column(TIME).to_list()
    return pl.DataFrame(
        {
            "policy_id": ["panel_dates"] * len(dates),
            "period": [value.isoformat() for value in dates],
            "requested_rebalance_date": dates,
            "rebalance_date": dates,
        },
        schema={
            "policy_id": pl.String,
            "period": pl.String,
            "requested_rebalance_date": pl.Date,
            "rebalance_date": pl.Date,
        },
    )


def _mapped_domain(schedule: pl.DataFrame, source: Panel) -> Domain:
    execution_dates = schedule.get_column("execution_date").unique().sort()
    return Domain(
        calendar=(execution_dates if len(execution_dates) else source.domain.times),
        universe=source.domain.asset_ids,
    )


def _open_sessions(calendar: pl.DataFrame) -> list[date]:
    if TIME not in calendar.columns:
        raise InputValidationError("calendar is missing required column: ['time']")
    sessions = calendar.with_columns(
        pl.col(TIME).cast(pl.Date, strict=False)
    ).drop_nulls(TIME)
    if "is_open" in sessions.columns:
        sessions = sessions.filter(pl.col("is_open").cast(pl.Int64) == 1)
    result = sessions.select(TIME).unique().sort(TIME).get_column(TIME).to_list()
    if not result:
        raise InputValidationError("calendar contains no open sessions")
    return result


def _observations(sessions: list[date], policy: AlphaPolicy) -> list[date]:
    if policy.anchor == EvaluationAnchor.EVERY_TRADING_DAY:
        return sessions
    months: dict[tuple[int, int], list[date]] = {}
    for session in sessions:
        months.setdefault((session.year, session.month), []).append(session)
    result: list[date] = []
    for (year, month), values in sorted(months.items()):
        if policy.anchor == EvaluationAnchor.LAST_TRADING_DAY:
            result.append(values[-1])
        elif policy.anchor == EvaluationAnchor.ON_OR_AFTER_CALENDAR_DAY:
            match = next(
                (value for value in values if value.day >= policy.calendar_day), None
            )
            if match is not None:
                result.append(match)
        elif policy.anchor == EvaluationAnchor.FIRST_WEEKDAY:
            anchor = date(year, month, 1)
            anchor = anchor.replace(
                day=1 + (int(policy.weekday) - anchor.weekday()) % 7
            )
            match = next((value for value in values if value >= anchor), None)
            if match is not None:
                result.append(match)
        elif policy.anchor == EvaluationAnchor.LAST_WEEKDAY:
            anchor = date(year, month, month_calendar.monthrange(year, month)[1])
            anchor = anchor.replace(
                day=anchor.day - (anchor.weekday() - int(policy.weekday)) % 7
            )
            match = next(
                (value for value in reversed(values) if value <= anchor), None
            )
            if match is not None:
                result.append(match)
    return result


__all__ = [
    "AlphaPolicy",
    "AlphaPolicyResult",
    "AlphaStandardization",
    "EvaluationAnchor",
    "EvaluationFrequency",
    "ExecutionPolicy",
    "HolidayAdjustment",
    "MissingSnapshotAction",
    "ScheduledPrediction",
    "ScheduledWeights",
    "alpha_policies",
    "execution_policies",
    "resolve_alpha_policy",
    "resolve_execution_policy",
]
