"""Standalone typed finite windows use the same numerical implementation."""

from dataclasses import replace

import polars as pl
import pytest
from bagelquant_core.hashing import hash_dataframe
from bagelquant_core.resources import ResourceLimits

from bagelquant_bt import (
    BTExecutionOptions,
    HorizonWork,
    compute_horizon_work,
    evaluate_alpha,
)
from test_incremental_alpha import SETTINGS, inputs


def test_standalone_dispatch_roundtrips_and_repeat_does_not_dispatch():
    alpha, labels, calendar, proof = inputs(300)
    options = dict(
        calendar=calendar,
        available_date=calendar["time"].max(),
        input_proof=proof,
        components=("horizons", "rolling_ic"),
        **SETTINGS,
    )
    reference = evaluate_alpha(alpha, labels, **options)

    class LocalDispatch:
        calls = 0

        def admit_horizons(self, working_bytes, retained_bytes, ready_units):
            assert working_bytes > 0 and retained_bytes > 0 and ready_units >= 2
            return 2

        def dispatch_horizons(self, work, *, progress):
            self.calls += 1
            results = []
            for unit in work:
                results.append(
                    compute_horizon_work(HorizonWork.from_dict(unit.to_dict()))
                )
                progress(len(results), len(work))
            return results

    dispatch = LocalDispatch()
    selected = BTExecutionOptions(
        workers=2,
        limits=ResourceLimits(total_threads=2, memory_target_mib=1024, cache_mib=32),
        window_dispatch=dispatch,
    )
    actual = evaluate_alpha(alpha, labels, options=selected, **options)
    assert actual.state == reference.state
    assert {name: hash_dataframe(frame) for name, frame in actual.tables.items()} == {
        name: hash_dataframe(frame) for name, frame in reference.tables.items()
    }
    repeated = evaluate_alpha(
        alpha, labels, previous=actual, options=selected, **options
    )
    assert repeated.execution["outcome"] == "up_to_date"
    assert repeated.tables is actual.tables
    assert dispatch.calls == 1


def test_worker_rejects_oversized_descriptor_before_reading_any_value(monkeypatch):
    from bagelquant_bt import SessionWindow

    work = HorizonWork(
        "/unused",
        SessionWindow("cumulative", "1", 1, 1),
        (),
        (),
        2,
        "cross_sectional_zscore",
        ResourceLimits(memory_target_mib=1, cache_mib=0),
        2 * 1024**2,
        1,
    )
    monkeypatch.setattr(
        pl, "read_ipc", lambda *_a, **_k: pytest.fail("read oversized work")
    )
    with pytest.raises(MemoryError, match="admitted memory"):
        compute_horizon_work(work)
    with pytest.raises(MemoryError):
        compute_horizon_work(replace(work, estimated_work_bytes=-1))


@pytest.mark.parametrize("damage", ["window", "date", "dtype"])
def test_caller_rejects_forged_child_scope_and_schema(damage):
    from pathlib import Path

    alpha, labels, calendar, proof = inputs(300)

    class CorruptDispatch:
        def admit_horizons(self, *_args):
            return 2

        def dispatch_horizons(self, work, *, progress):
            results = [compute_horizon_work(unit) for unit in work]
            unit, result = work[0], results[0]
            item = next(frame for frame in result.frames if frame.name == "ic")
            path = Path(unit.staging_root) / item.filename
            frame = pl.read_ipc(path, memory_map=False)
            if damage == "window":
                frame = frame.with_columns(pl.lit("foreign").alias("window_id"))
            elif damage == "date":
                from datetime import timedelta

                frame = frame.with_columns(
                    pl.col("evaluation_date") + timedelta(days=10000)
                )
            else:
                frame = frame.with_columns(pl.col("pearson_ic").cast(pl.Float32))
                changed = replace(
                    item,
                    schema=tuple(
                        (key, str(dtype)) for key, dtype in frame.schema.items()
                    ),
                )
                results[0] = replace(
                    result,
                    frames=tuple(
                        changed if x.name == "ic" else x for x in result.frames
                    ),
                )
            frame.write_ipc(path)
            return results

    with pytest.raises(ValueError, match=r"another window|calendar|family schema"):
        evaluate_alpha(
            alpha,
            labels,
            calendar=calendar,
            available_date=calendar["time"].max(),
            input_proof=proof,
            components=("horizons",),
            options=BTExecutionOptions(
                workers=2,
                limits=ResourceLimits(
                    total_threads=2, memory_target_mib=1024, cache_mib=16
                ),
                window_dispatch=CorruptDispatch(),
            ),
            **SETTINGS,
        )
