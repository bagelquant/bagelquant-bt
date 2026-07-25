"""Signal sampling policies with explicit observation and execution dates."""
# ruff: noqa: E501

from __future__ import annotations

import calendar as month_calendar
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Literal

import polars as pl

from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME, validate_panel_frame


class SignalFrequency(StrEnum):
    DAILY = "daily"
    MONTHLY = "monthly"


class SignalAnchor(StrEnum):
    EVERY_TRADING_DAY = "every_trading_day"
    LAST_TRADING_DAY = "last_trading_day"
    ON_OR_AFTER_CALENDAR_DAY = "on_or_after_calendar_day"
    FIRST_WEEKDAY = "first_weekday"
    LAST_WEEKDAY = "last_weekday"


class HolidayAdjustment(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    NEXT_OPEN_SESSION = "next_open_session"
    PREVIOUS_OPEN_SESSION = "previous_open_session"


@dataclass(frozen=True, slots=True)
class SignalPolicy:
    """Choose prediction snapshots and make them executable on open sessions."""

    id: str
    frequency: Literal["daily", "monthly"]
    anchor: SignalAnchor
    calendar_day: int | None = None
    weekday: int | None = None
    observation_offset_sessions: int = 0
    execution_lag_sessions: int = 1
    holiday_adjustment: HolidayAdjustment = HolidayAdjustment.NOT_APPLICABLE

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("signal policy requires an id")
        if self.observation_offset_sessions < 0:
            raise ValueError("observation offset cannot be negative")
        if self.execution_lag_sessions <= 0:
            raise ValueError("execution lag must be positive")
        if self.anchor == SignalAnchor.ON_OR_AFTER_CALENDAR_DAY:
            if self.calendar_day is None or not 1 <= self.calendar_day <= 28:
                raise ValueError("calendar-day anchor requires a day from 1 through 28")
        elif self.calendar_day is not None:
            raise ValueError("calendar_day is valid only for on-or-after anchor")
        if self.anchor in {SignalAnchor.FIRST_WEEKDAY, SignalAnchor.LAST_WEEKDAY}:
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
        """Resolve policy dates from an explicit exchange calendar."""

        sessions = _open_sessions(calendar)
        rows: list[dict[str, object]] = []
        positions = {session: index for index, session in enumerate(sessions)}
        for anchored in _observations(sessions, self):
            observation_position = positions[anchored] - self.observation_offset_sessions
            if observation_position < 0:
                continue
            observation = sessions[observation_position]
            execution_position = observation_position + self.execution_lag_sessions
            if execution_position >= len(sessions):
                continue
            if (start is not None and observation < start) or (end is not None and observation > end):
                continue
            rows.append({
                "policy_id": self.id,
                "period": observation.isoformat() if self.frequency == "daily" else observation.strftime("%Y-%m"),
                "observation_time": observation,
                TIME: sessions[execution_position],
            })
        return pl.DataFrame(rows, schema={
            "policy_id": pl.String, "period": pl.String,
            "observation_time": pl.Date, TIME: pl.Date,
        }).sort("observation_time")

    def transform(self, predictions: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
        """Return executable signals from already-filled prediction snapshots."""

        values = validate_panel_frame(predictions, label="predictions", value_columns=("prediction",))
        schedule = self.schedule(calendar)
        if schedule.is_empty():
            return pl.DataFrame(schema={"observation_time": pl.Date, TIME: pl.Date, ASSET_ID: pl.String, "signal": pl.Float64})
        return (
            schedule.join(values, left_on="observation_time", right_on=TIME, how="left")
            .drop_nulls("prediction")
            .select("observation_time", TIME, ASSET_ID, pl.col("prediction").alias("signal"))
            .sort([TIME, ASSET_ID])
        )


_CANONICAL_SIGNAL_POLICIES = {
    "daily": SignalPolicy("daily", "daily", SignalAnchor.EVERY_TRADING_DAY),
    "month_end": SignalPolicy("month_end", "monthly", SignalAnchor.LAST_TRADING_DAY),
    "monthly_mid": SignalPolicy("monthly_mid", "monthly", SignalAnchor.ON_OR_AFTER_CALENDAR_DAY, calendar_day=15, holiday_adjustment=HolidayAdjustment.NEXT_OPEN_SESSION),
    "monthly_first_monday": SignalPolicy("monthly_first_monday", "monthly", SignalAnchor.FIRST_WEEKDAY, weekday=0, holiday_adjustment=HolidayAdjustment.NEXT_OPEN_SESSION),
    "monthly_last_friday": SignalPolicy("monthly_last_friday", "monthly", SignalAnchor.LAST_WEEKDAY, weekday=4, holiday_adjustment=HolidayAdjustment.PREVIOUS_OPEN_SESSION),
}


def signal_policies() -> tuple[SignalPolicy, ...]:
    return tuple(_CANONICAL_SIGNAL_POLICIES.values())


def resolve_signal_policy(policy_id: str) -> SignalPolicy:
    try:
        return _CANONICAL_SIGNAL_POLICIES[policy_id]
    except KeyError as error:
        raise KeyError(f"unknown signal policy: {policy_id}") from error


def _open_sessions(calendar: pl.DataFrame) -> list[date]:
    if TIME not in calendar.columns:
        raise InputValidationError("calendar is missing required column: ['time']")
    sessions = calendar.with_columns(pl.col(TIME).cast(pl.Date, strict=False)).drop_nulls(TIME)
    if "is_open" in sessions.columns:
        sessions = sessions.filter(pl.col("is_open").cast(pl.Int64) == 1)
    result = sessions.select(TIME).unique().sort(TIME).get_column(TIME).to_list()
    if not result:
        raise InputValidationError("calendar contains no open sessions")
    return result


def _observations(sessions: list[date], policy: SignalPolicy) -> list[date]:
    if policy.anchor == SignalAnchor.EVERY_TRADING_DAY:
        return sessions
    months: dict[tuple[int, int], list[date]] = {}
    for session in sessions:
        months.setdefault((session.year, session.month), []).append(session)
    result: list[date] = []
    for (year, month), values in sorted(months.items()):
        if policy.anchor == SignalAnchor.LAST_TRADING_DAY:
            result.append(values[-1])
        elif policy.anchor == SignalAnchor.ON_OR_AFTER_CALENDAR_DAY:
            match = next((value for value in values if value.day >= policy.calendar_day), None)
            if match is not None:
                result.append(match)
        elif policy.anchor == SignalAnchor.FIRST_WEEKDAY:
            anchor = date(year, month, 1)
            anchor = anchor.replace(day=1 + (int(policy.weekday) - anchor.weekday()) % 7)
            match = next((value for value in values if value >= anchor), None)
            if match is not None:
                result.append(match)
        elif policy.anchor == SignalAnchor.LAST_WEEKDAY:
            anchor = date(year, month, month_calendar.monthrange(year, month)[1])
            anchor = anchor.replace(day=anchor.day - (anchor.weekday() - int(policy.weekday)) % 7)
            match = next((value for value in reversed(values) if value <= anchor), None)
            if match is not None:
                result.append(match)
    return result


__all__ = [
    "HolidayAdjustment", "SignalAnchor", "SignalFrequency", "SignalPolicy",
    "resolve_signal_policy", "signal_policies",
]
