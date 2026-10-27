"""Metadata-only admission of bounded causal risk-profile updates."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

RISK_PROFILE_POLICY = "rolling_risk_profile.v3"


def risk_profile_policy(
    regression_window: int, volatility_window: int, annualization: int, ridge: float,
    regression_min_periods: int | None = None,
) -> dict[str, Any]:
    if (
        isinstance(regression_window, bool)
        or not isinstance(regression_window, int)
        or regression_window < 2
        or isinstance(volatility_window, bool)
        or not isinstance(volatility_window, int)
        or volatility_window < 2
        or isinstance(annualization, bool)
        or not isinstance(annualization, int)
        or annualization < 1
    ):
        raise ValueError("invalid risk profile windows or annualization")
    minimum = (
        regression_window if regression_min_periods is None else regression_min_periods
    )
    if (isinstance(minimum, bool) or not isinstance(minimum, int)
            or not 2 <= minimum <= regression_window):
        raise ValueError(
            "risk profile regression_min_periods must be between 2 "
            "and regression_window"
        )
    import math

    if isinstance(ridge, bool) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("invalid risk profile ridge penalty")
    return {
        "kernel": RISK_PROFILE_POLICY,
        "regression_window": regression_window,
        "regression_min_periods": minimum,
        "volatility_window": volatility_window,
        "annualization": annualization,
        "ridge": float(ridge),
    }


@dataclass(frozen=True, slots=True)
class RiskProfileUpdatePlan:
    """A numerical output interval and its input/residual halos.

    ``previous_index`` is -1 for the supplied current state, or an index into
    ``preceding``. Only metadata is consumed by planning. A repair requires an
    original preceding state whose source prefix is still proved equal.
    """

    mode: str
    calendar: tuple[date, ...]
    through: date
    policy: Mapping[str, Any]
    read_start: date | None = None
    output_start: date | None = None
    output_end: date | None = None
    specific_start: date | None = None
    previous_index: int | None = None
    reason: str | None = None
    retained_state: Mapping[str, Any] | None = None

    @property
    def complete(self):
        return self.mode == "up_to_date"

    @property
    def read_end(self):
        return self.output_end


def plan_risk_profile_update(
    previous: Mapping[str, Any] | None,
    *,
    calendar: Sequence[date],
    through: date,
    affected_intervals: Sequence[tuple[date, date]] = (),
    preceding: Sequence[Mapping[str, Any]] = (),
    input_proof: Callable[[date, date, date | None], str | None] | None = None,
    regression_window: int = 240,
    regression_min_periods: int | None = None,
    volatility_window: int = 120,
    annualization: int = 240,
    ridge: float = 1.0,
) -> RiskProfileUpdatePlan:
    """Plan initialization/append/repair without reading numerical history.

    Proofs bind gross returns, factor population/families/availability and the
    complete original calendar, including empty/null coordinates. Missing safe
    state or unknown changes block continuation instead of replaying history.
    """
    sessions = tuple(calendar)
    if (
        not sessions
        or any(type(day) is not date for day in sessions)
        or tuple(sorted(set(sessions))) != sessions
        or type(through) is not date
        or through < sessions[0]
    ):
        raise ValueError("risk profile planning needs a sorted nonempty date calendar")
    days = sessions[: bisect_right(sessions, through)]
    policy = risk_profile_policy(
        regression_window, volatility_window, annualization, ridge,
        regression_min_periods,
    )

    def result(mode, *, index=None, output=None, end=None, reason=None, retained=None):
        if output is None:
            return RiskProfileUpdatePlan(
                mode, sessions, through, policy, previous_index=index, reason=reason
            )
        position = bisect_left(days, output)
        return RiskProfileUpdatePlan(
            mode,
            sessions,
            through,
            policy,
            read_start=days[
                max(0, position - max(regression_window, volatility_window - 1))
            ],
            output_start=output,
            output_end=end or days[-1],
            specific_start=days[max(0, position - volatility_window + 1)],
            previous_index=index,
            reason=reason,
            retained_state=retained,
        )

    if previous is None:
        return result("initialize", output=days[0])

    def admitted(state):
        try:
            boundary = date.fromisoformat(state["through"])
            cutoff = date.fromisoformat(state["cutoff"])
            return (
                state["policy"] == policy
                and state["calendar_start"] == str(days[0])
                and boundary in days
                and cutoff <= through
                and state["input_prefix"] is not None
                and isinstance(state["factors"], dict)
                and "pending_output_start" in state
                and "proof_through" in state
                and input_proof is not None
                and input_proof(
                    days[0], date.fromisoformat(state["proof_through"]), cutoff
                )
                == state["input_prefix"]
            )
        except (KeyError, ValueError, TypeError):
            return False

    changes = []
    for first, last in affected_intervals:
        if type(first) is not date or type(last) is not date or first > last:
            raise ValueError("risk profile affected intervals need ordered dates")
        if first <= days[-1] and last >= days[0]:
            changes.append((max(first, days[0]), min(last, days[-1])))
    if changes:
        try:
            if date.fromisoformat(previous["through"]) != days[-1] or previous[
                "cutoff"
            ] != str(through):
                return result(
                    "blocked",
                    reason=(
                        "risk profile repair and append need separate "
                        "admitted output plans"
                    ),
                )
        except (KeyError, ValueError):
            return result(
                "blocked",
                reason="risk profile repair needs original completion metadata",
            )
        affected = days[
            min(bisect_left(days, min(first for first, _ in changes)), len(days) - 1)
        ]
        last_changed = bisect_right(days, max(last for _, last in changes)) - 1
        end = days[
            min(last_changed + regression_window + volatility_window - 1, len(days) - 1)
        ]
        candidates = [(-1, previous), *enumerate(preceding)]
        candidates.sort(key=lambda item: item[1].get("through", ""), reverse=True)
        for index, state in candidates:
            if admitted(state) and date.fromisoformat(state["through"]) < affected:
                pending = previous["pending_output_start"]
                output = (
                    min(affected, date.fromisoformat(pending)) if pending else affected
                )
                return result(
                    "repair", index=index, output=output, end=end, retained=previous
                )
        return result(
            "blocked",
            reason=(
                "affected risk profile inputs lack a proven preceding checkpoint; "
                "retain original results and explicitly restore original state"
            ),
        )
    if not admitted(previous):
        return result(
            "blocked",
            reason=(
                "risk profile continuation needs matching policy/state and original "
                "source-prefix evidence; explicitly maintain or plan affected inputs"
            ),
        )
    boundary = date.fromisoformat(previous["through"])
    if boundary == days[-1] and previous["cutoff"] == str(through):
        return result("up_to_date", index=-1)
    position = min(bisect_right(days, boundary), len(days) - 1)
    output = days[position]
    if previous["pending_output_start"]:
        output = min(output, date.fromisoformat(previous["pending_output_start"]))
    return result("append", index=-1, output=output)
