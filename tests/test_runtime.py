"""Explicit local worker ceilings and deterministic batch ordering."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest
from bagelquant_core.resources import ResourceLimits, active_resource_limits

from bagelquant_bt.runtime import BTExecutionOptions, run_evaluation_batch


def test_default_execution_is_serial_and_does_not_probe_hardware(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: (_ for _ in ()).throw(AssertionError()))
    observed = []
    result = run_evaluation_batch(range(4), lambda i: (observed.append(i), i * i)[1])
    assert observed == [0, 1, 2, 3]
    assert result == [0, 1, 4, 9]


def test_later_completed_failure_stops_admission_before_ordered_wait_releases(
    monkeypatch,
):
    failed = Event()
    admitted = []

    class ObservedExecutor(ThreadPoolExecutor):
        def submit(self, fn, item):
            future = super().submit(fn, item)
            if item == 1:
                future.add_done_callback(lambda _: failed.set())
            return future

    def items():
        for item in range(3):
            admitted.append(item)
            yield item

    def evaluate(item):
        if item == 0:
            assert failed.wait(5)
            return item
        raise ValueError("later failure")

    monkeypatch.setattr("bagelquant_bt.runtime.ThreadPoolExecutor", ObservedExecutor)
    with pytest.raises(ValueError, match="later failure"):
        run_evaluation_batch(
            items(),
            evaluate,
            options=BTExecutionOptions(
                workers=2, limits=ResourceLimits(total_threads=2)
            ),
        )
    assert admitted == [0, 1]


def test_parallel_admission_is_bounded_and_results_keep_input_order():
    lock = Lock()
    overlap = Event()
    active = 0
    peak = 0
    budgets = []

    def evaluate(item):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            budgets.append(active_resource_limits())
            if active == 2:
                overlap.set()
        assert overlap.wait(5)
        with lock:
            active -= 1
        return item * 2

    limits = ResourceLimits(
        total_threads=4,
        lightgbm_threads=4,
        memory_target_mib=100,
        cache_mib=20,
        batch_rows=100,
    )
    assert run_evaluation_batch(
        range(5), evaluate, options=BTExecutionOptions(workers=2, limits=limits)
    ) == [0, 2, 4, 6, 8]
    assert peak == 2
    assert all(
        b.total_threads == 2 and b.memory_target_mib == 50 and b.batch_rows == 50
        for b in budgets
    )


def test_failure_stops_later_admission_and_serial_cancel_stops_work():
    visited = []

    def evaluate(item):
        visited.append(item)
        if item == 0:
            raise ValueError("failed")
        return item

    limits = ResourceLimits(total_threads=2)
    with pytest.raises(ValueError, match="failed"):
        run_evaluation_batch(
            range(20), evaluate, options=BTExecutionOptions(workers=2, limits=limits)
        )
    assert set(visited) <= {0, 1}

    def cancel():
        if len(visited) >= 2:
            raise InterruptedError("cancel")

    visited.clear()
    with pytest.raises(InterruptedError):
        run_evaluation_batch(
            range(20), lambda i: (visited.append(i), i)[1], check_canceled=cancel
        )
    assert visited == [0, 1]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, 3])
def test_workers_must_fit_explicit_budget(workers):
    with pytest.raises(ValueError):
        BTExecutionOptions(workers=workers, limits=ResourceLimits(total_threads=2))
