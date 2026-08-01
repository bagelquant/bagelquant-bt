"""Signal snapshot selection and execution scheduling."""

from __future__ import annotations

import calendar as month_calendar
import hashlib
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Literal

import polars as pl
from bagelquant_core import Domain, SignalPanel

from .exceptions import InputValidationError
from .inputs import ASSET_ID, TIME


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


class MissingSnapshotAction(StrEnum):
    SKIP = "skip"
    PREVIOUS_IN_PERIOD = "previous_in_period"


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Map a rebalance session to a later executable market session."""

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
        """Attach an execution date counted from each rebalance date."""

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


@dataclass(frozen=True, slots=True)
class ScheduledSignal:
    """A resolved schedule and its strongly typed executable SignalPanel."""

    schedule: pl.DataFrame
    signal: SignalPanel

    @property
    def identity(self) -> str:
        """Return a deterministic identity shared by every downstream consumer."""

        payload = self.schedule.sort("requested_rebalance_date").write_json()
        return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


@dataclass(frozen=True, slots=True)
class SignalDatePolicy:
    """Choose Alpha snapshots without deciding when orders execute."""

    id: str
    frequency: Literal["daily", "monthly"]
    anchor: SignalAnchor
    missing_snapshot: MissingSnapshotAction
    calendar_day: int | None = None
    weekday: int | None = None
    observation_offset_sessions: int = 0
    holiday_adjustment: HolidayAdjustment = HolidayAdjustment.NOT_APPLICABLE

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("signal policy requires an id")
        if self.observation_offset_sessions < 0:
            raise ValueError("observation offset cannot be negative")
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
        """Resolve requested rebalance sessions from an exchange calendar."""

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

    def select(
        self,
        predictions: SignalPanel,
        calendar: pl.DataFrame,
        *,
        execution_policy: str | ExecutionPolicy = "next_open",
        start: date | None = None,
        end: date | None = None,
    ) -> ScheduledSignal:
        """Select whole Alpha snapshots, then map rebalances to execution dates."""

        if not isinstance(predictions, SignalPanel):
            raise TypeError("SignalDatePolicy.select requires a SignalPanel")
        values = predictions.collect(dense=False).drop_nulls("value")
        requested = self.schedule(calendar, start=start, end=end)
        execution = (
            resolve_execution_policy(execution_policy)
            if isinstance(execution_policy, str)
            else execution_policy
        )
        requested = execution.resolve(requested, calendar)
        available_dates = (
            values.select(pl.col(TIME).alias("alpha_date")).unique().sort("alpha_date")
        )
        schedule_rows: list[dict[str, object]] = []
        signal_frames: list[pl.DataFrame] = []
        for row in requested.iter_rows(named=True):
            rebalance = row["rebalance_date"]
            period = row["period"]
            candidates = available_dates.filter(pl.col("alpha_date") <= rebalance)
            if self.frequency == "monthly":
                candidates = candidates.filter(
                    pl.col("alpha_date").dt.strftime("%Y-%m") == period
                )
            exact = candidates.filter(pl.col("alpha_date") == rebalance)
            selected_date = (
                exact.get_column("alpha_date").max()
                if not exact.is_empty()
                else (
                    candidates.get_column("alpha_date").max()
                    if self.missing_snapshot
                    == MissingSnapshotAction.PREVIOUS_IN_PERIOD
                    and not candidates.is_empty()
                    else None
                )
            )
            schedule_row = {
                **row,
                "alpha_date": selected_date,
                "selection_status": (
                    "exact"
                    if selected_date == rebalance
                    else "previous_in_period"
                    if selected_date is not None
                    else "skipped"
                ),
                "skip_reason": (
                    None if selected_date is not None else "missing_alpha_snapshot"
                ),
            }
            if row["execution_date"] is None:
                schedule_row["selection_status"] = "skipped"
                schedule_row["skip_reason"] = "missing_execution_session"
            schedule_rows.append(schedule_row)
            if (
                selected_date is None
                or schedule_row["selection_status"] == "skipped"
            ):
                continue
            signal_frames.append(
                values.filter(pl.col(TIME) == selected_date).select(
                    pl.lit(self.id).alias("policy_id"),
                    pl.lit(execution.id).alias("execution_policy_id"),
                    pl.lit(period).alias("period"),
                    pl.lit(selected_date, dtype=pl.Date).alias("alpha_date"),
                    pl.lit(rebalance, dtype=pl.Date).alias("rebalance_date"),
                    pl.lit(row["execution_date"], dtype=pl.Date).alias(
                        "execution_date"
                    ),
                    pl.col(TIME).alias("source_time"),
                    pl.lit(row["execution_date"], dtype=pl.Date).alias(TIME),
                    ASSET_ID,
                    pl.col("value").alias("value"),
                )
            )
        schedule = pl.DataFrame(
            schedule_rows,
            schema={
                "policy_id": pl.String,
                "period": pl.String,
                "requested_rebalance_date": pl.Date,
                "rebalance_date": pl.Date,
                "execution_policy_id": pl.String,
                "execution_date": pl.Date,
                "alpha_date": pl.Date,
                "selection_status": pl.String,
                "skip_reason": pl.String,
            },
        ).sort("rebalance_date")
        signals = (
            pl.concat(signal_frames).sort(TIME, ASSET_ID)
            if signal_frames
            else _empty_signals()
        )
        execution_dates = (
            requested.get_column("execution_date").drop_nulls().unique().sort()
        )
        calendar_dates = (
            execution_dates
            if not execution_dates.is_empty()
            else predictions.domain.times
        )
        scheduled_domain = Domain(
            calendar=calendar_dates,
            universe=predictions.domain.asset_ids,
        )
        scheduled_panel = SignalPanel.from_domain(
            signals.select(TIME, ASSET_ID, "value"),
            scheduled_domain,
            name=predictions.name,
            metadata={
                **predictions.metadata,
                "signal_date_policy": self.id,
                "execution_policy": execution.id,
            },
        )
        return ScheduledSignal(schedule=schedule, signal=scheduled_panel)

    def transform(
        self,
        predictions: SignalPanel,
        calendar: pl.DataFrame,
        *,
        execution_policy: str | ExecutionPolicy = "next_open",
    ) -> pl.DataFrame:
        """Return executable rows; prefer :meth:`select` when lineage is needed."""

        return self.select(
            predictions, calendar, execution_policy=execution_policy
        ).signal.collect(dense=False)


_CANONICAL_EXECUTION_POLICIES = {
    "next_open": ExecutionPolicy("next_open", lag_sessions=1),
}

_CANONICAL_SIGNAL_DATE_POLICIES = {
    "daily": SignalDatePolicy(
        "daily",
        "daily",
        SignalAnchor.EVERY_TRADING_DAY,
        MissingSnapshotAction.SKIP,
    ),
    "month_end": SignalDatePolicy(
        "month_end",
        "monthly",
        SignalAnchor.LAST_TRADING_DAY,
        MissingSnapshotAction.PREVIOUS_IN_PERIOD,
    ),
    "monthly_mid": SignalDatePolicy(
        "monthly_mid",
        "monthly",
        SignalAnchor.ON_OR_AFTER_CALENDAR_DAY,
        MissingSnapshotAction.SKIP,
        calendar_day=15,
        holiday_adjustment=HolidayAdjustment.NEXT_OPEN_SESSION,
    ),
    "monthly_first_monday": SignalDatePolicy(
        "monthly_first_monday",
        "monthly",
        SignalAnchor.FIRST_WEEKDAY,
        MissingSnapshotAction.SKIP,
        weekday=0,
        holiday_adjustment=HolidayAdjustment.NEXT_OPEN_SESSION,
    ),
    "monthly_last_friday": SignalDatePolicy(
        "monthly_last_friday",
        "monthly",
        SignalAnchor.LAST_WEEKDAY,
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


def signal_date_policies() -> tuple[SignalDatePolicy, ...]:
    return tuple(_CANONICAL_SIGNAL_DATE_POLICIES.values())


def resolve_signal_date_policy(policy_id: str) -> SignalDatePolicy:
    try:
        return _CANONICAL_SIGNAL_DATE_POLICIES[policy_id]
    except KeyError as error:
        raise KeyError(f"unknown signal date policy: {policy_id}") from error


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


def _observations(sessions: list[date], policy: SignalDatePolicy) -> list[date]:
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
            match = next(
                (value for value in values if value.day >= policy.calendar_day), None
            )
            if match is not None:
                result.append(match)
        elif policy.anchor == SignalAnchor.FIRST_WEEKDAY:
            anchor = date(year, month, 1)
            anchor = anchor.replace(
                day=1 + (int(policy.weekday) - anchor.weekday()) % 7
            )
            match = next((value for value in values if value >= anchor), None)
            if match is not None:
                result.append(match)
        elif policy.anchor == SignalAnchor.LAST_WEEKDAY:
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


def _empty_signals() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "policy_id": pl.String,
            "execution_policy_id": pl.String,
            "period": pl.String,
            "alpha_date": pl.Date,
            "rebalance_date": pl.Date,
            "execution_date": pl.Date,
            "source_time": pl.Date,
            TIME: pl.Date,
            ASSET_ID: pl.String,
            "value": pl.Float64,
        }
    )


__all__ = [
    "ExecutionPolicy",
    "HolidayAdjustment",
    "MissingSnapshotAction",
    "ScheduledSignal",
    "SignalAnchor",
    "SignalDatePolicy",
    "SignalFrequency",
    "execution_policies",
    "resolve_execution_policy",
    "resolve_signal_date_policy",
    "signal_date_policies",
]
