"""Typed, computation-only horizon work on caller-owned temporary Arrow files.

BT owns the numerical kernel and validates staged results. The caller owns
process admission and lifetime; workers never open stores or publish receipts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from typing import Protocol
from uuid import uuid4

import polars as pl
from bagelquant_core.resources import ResourceLimits, resource_limits

from .horizon import SessionWindow

FAMILIES = (
    "coverage",
    "ic",
    "book_returns",
    "tail_returns",
    "quantile_forward_returns",
    "quantile_structure",
    "factor_returns",
)


@dataclass(frozen=True, slots=True)
class HorizonFrame:
    name: str
    filename: str
    rows: int
    schema: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, body):
        return cls(**{**body, "schema": tuple(tuple(item) for item in body["schema"])})


@dataclass(frozen=True, slots=True)
class HorizonWork:
    """One independently computable window with an explicit local memory bound."""

    staging_root: str
    window: SessionWindow
    inputs: tuple[HorizonFrame, ...]
    previous: tuple[HorizonFrame, ...]
    quantiles: int
    factor_standardization: str
    limits: ResourceLimits
    estimated_work_bytes: int
    max_output_rows: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, body: dict) -> HorizonWork:
        return cls(
            **{
                **body,
                "window": SessionWindow(**body["window"]),
                "limits": ResourceLimits(**body["limits"]),
                "inputs": tuple(
                    HorizonFrame.from_dict(item) for item in body["inputs"]
                ),
                "previous": tuple(
                    HorizonFrame.from_dict(item) for item in body["previous"]
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class HorizonResult:
    """Bounded output descriptors and identity-neutral worker measurements."""

    window: SessionWindow
    frames: tuple[HorizonFrame, ...]
    cpu_seconds: float
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, body: dict) -> HorizonResult:
        return cls(
            **{
                **body,
                "window": SessionWindow(**body["window"]),
                "frames": tuple(
                    HorizonFrame.from_dict(item) for item in body["frames"]
                ),
            }
        )


class HorizonDispatch(Protocol):
    """Caller process policy, separate from BT's formulas and artifact identity."""

    def admit_horizons(
        self, working_bytes: int, retained_bytes: int, ready_units: int
    ) -> int:
        """Freeze complete parent/child admission; return one for inline work."""
        ...

    def dispatch_horizons(
        self, work: Sequence[HorizonWork], *, progress: Callable[[int, int], None]
    ) -> Sequence[HorizonResult]:
        """Run the admitted batch, draining children before returning or raising."""
        ...


def _path(root: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("Horizon staging file must be a direct child")
    path = root / filename
    if (
        root.is_symlink()
        or path.is_symlink()
        or path.resolve().parent != root.resolve()
    ):
        raise ValueError("Horizon staging path escapes its root")
    return path


def _write(root, name, frame):
    filename = f"{uuid4().hex}.arrow"
    frame.write_ipc(_path(root, filename), compression="uncompressed")
    return HorizonFrame(
        name,
        filename,
        frame.height,
        tuple((column, str(dtype)) for column, dtype in frame.schema.items()),
    )


def _read(root, descriptor, *, memory_map=False):
    frame = pl.read_ipc(_path(root, descriptor.filename), memory_map=memory_map)
    if (
        frame.height != descriptor.rows
        or tuple((name, str(dtype)) for name, dtype in frame.schema.items())
        != descriptor.schema
    ):
        raise ValueError("Horizon staged frame differs from its descriptor")
    return frame


def compute_horizon_work(work: HorizonWork) -> HorizonResult:
    """Compute the same private window kernel used inline, without publication."""
    from .evaluation import _alpha_window_frames

    if (
        work.estimated_work_bytes < 0
        or work.max_output_rows < 0
        or work.estimated_work_bytes > work.limits.memory_target_mib * 1024**2
    ):
        raise MemoryError("Horizon work exceeds its admitted memory slice")
    started, cpu = perf_counter(), process_time()
    root = Path(work.staging_root)
    with resource_limits(work.limits):
        # Shared uncompressed Arrow inputs stay immutable until child work
        # drains. Map their buffers instead of copying the whole common panel
        # into each window invocation. Returned parent frames are copied before
        # the staging context expires.
        inputs = {item.name: _read(root, item, memory_map=True) for item in work.inputs}
        previous = {
            item.name: _read(root, item, memory_map=True) for item in work.previous
        }
        frames = _alpha_window_frames(
            work.window,
            **inputs,
            previous=previous,
            quantiles=work.quantiles,
            factor_standardization=work.factor_standardization,
            check_canceled=lambda: None,
        )
        if sum(frame.height for frame in frames.values()) > work.max_output_rows:
            raise MemoryError("Horizon result exceeds its admitted output bound")
        output = tuple(_write(root, name, frames[name]) for name in FAMILIES)
    return HorizonResult(
        work.window, output, process_time() - cpu, perf_counter() - started
    )


@contextmanager
def staged_horizon_batch(
    windows,
    frames: Mapping[str, pl.DataFrame],
    *,
    previous,
    quantiles,
    factor_standardization,
    limits,
    working_bytes,
    check_canceled: Callable[[], None],
):
    """Stage shared inputs once and private retained windows after admission."""
    with TemporaryDirectory(prefix="bagelquant-horizons-") as temporary:
        root = Path(temporary)
        inputs = []
        for name, frame in frames.items():
            check_canceled()
            inputs.append(_write(root, name, frame))
        work = []
        for window in windows:
            check_canceled()
            old = {} if previous is None else previous(window)
            staged = tuple(_write(root, name, frame) for name, frame in old.items())
            bound = frames["sessions"].height * (quantiles + 2) * 16
            bound += sum(frame.height for frame in old.values())
            work.append(
                HorizonWork(
                    str(root),
                    window,
                    tuple(inputs),
                    staged,
                    quantiles,
                    factor_standardization,
                    limits,
                    working_bytes,
                    bound,
                )
            )
        yield tuple(work)


def read_horizon_result(
    work: HorizonWork, result: HorizonResult
) -> dict[str, pl.DataFrame]:
    """Validate the selected window, output scope and typed descriptors on return."""
    if (
        result.window != work.window
        or tuple(item.name for item in result.frames) != FAMILIES
        or any(item.rows < 0 for item in result.frames)
        or sum(item.rows for item in result.frames) > work.max_output_rows
    ):
        raise ValueError("Horizon result differs from its admitted window/output scope")
    frames = {item.name: _read(Path(work.staging_root), item) for item in result.frames}
    _validate_result_frames(work, frames)
    return frames


def _validate_result_frames(work, frames):
    """Validate BT's family contract independently of child-supplied headers."""
    dates = _read(
        Path(work.staging_root),
        next(item for item in work.inputs if item.name == "sessions"),
    ).select(pl.col("time").alias("evaluation_date"))
    common = {
        "evaluation_date": pl.Date,
        "execution_date": pl.Date,
        "target_end_date": pl.Date,
        "available_date": pl.Date,
        "window_kind": pl.String,
        "window_id": pl.String,
        "start_session": pl.Int64,
        "end_session": pl.Int64,
    }
    counts = {
        "expected_count": pl.UInt32,
        "observed_count": pl.UInt32,
        "coverage_ratio": pl.Float64,
        "unavailable_reason": pl.String,
    }
    schemas = {
        "coverage": {
            **counts,
            "expected_count": pl.Int64,
            "observed_count": pl.Int64,
            "is_mature": pl.Boolean,
            "is_complete": pl.Boolean,
        },
        "ic": {
            "pearson_ic": pl.Float64,
            "spearman_ic": pl.Float64,
            "sample_size": pl.Int64,
        },
        "book_returns": {**counts, "book_return": pl.Float64},
        "tail_returns": {**counts, "tail_return": pl.Float64},
        "quantile_forward_returns": {
            **counts,
            "quantile": pl.String,
            "quantile_return": pl.Float64,
        },
        "quantile_structure": {
            "quantile_rank_ic": pl.Float64,
            "monotonicity": pl.Float64,
            "quantile_linearity_slope": pl.Float64,
            "quantile_linearity_r_squared": pl.Float64,
            "unavailable_reason": pl.String,
        },
        "factor_returns": {"factor_return": pl.Float64, "sample_size": pl.UInt32},
    }
    scope = {
        "window_kind": work.window.window_kind,
        "window_id": work.window.window_id,
        "start_session": work.window.start_session,
        "end_session": work.window.end_session,
    }
    for name, frame in frames.items():
        expected = {**common, **schemas[name]}
        if "target_start_date" in frame.columns:
            expected["target_start_date"] = pl.Date
        if dict(frame.schema) != expected:
            raise ValueError(f"Horizon {name} has an invalid typed family schema")
        if any(
            frame[column].null_count() or (frame[column] != value).any()
            for column, value in scope.items()
        ):
            raise ValueError(f"Horizon {name} rows belong to another window")
        if (
            frame["evaluation_date"].null_count()
            or frame.select("evaluation_date")
            .unique()
            .join(dates, on="evaluation_date", how="anti")
            .height
        ):
            raise ValueError(f"Horizon {name} rows are outside the admitted calendar")
        keys = [
            "evaluation_date",
            *(["quantile"] if name == "quantile_forward_returns" else []),
        ]
        if frame.select(keys).is_duplicated().any():
            raise ValueError(f"Horizon {name} has duplicate output coordinates")
