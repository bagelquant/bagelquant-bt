"""Explicit bounded local execution; global admission remains with callers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

from bagelquant_core.resources import ResourceLimits, resource_limits


@dataclass(frozen=True, slots=True)
class BTExecutionOptions:
    """Caller-supplied worker, thread, memory and batch ceilings.

    Resource limits are execution-only and never enter result identity. Kernels
    receive a divided Core budget; this module never probes machine resources.
    """

    workers: int = 1
    limits: ResourceLimits = field(
        default_factory=lambda: ResourceLimits(total_threads=1, lightgbm_threads=1)
    )

    native_admission: Callable[[str], AbstractContextManager] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.native_admission is not None and not callable(self.native_admission):
            raise ValueError("native_admission must be callable")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int):
            raise ValueError("workers must be a positive integer")
        if self.workers < 1 or self.workers > self.limits.total_threads:
            raise ValueError("workers must fit the supplied total thread budget")
        if self.workers > self.limits.memory_target_mib:
            raise ValueError("workers must fit the supplied memory budget")


def run_evaluation_batch[Input, Output](
    items: Iterable[Input],
    evaluate: Callable[[Input], Output],
    *,
    options: BTExecutionOptions | None = None,
    check_canceled: Callable[[], None] = lambda: None,
    progress: Callable[[int, int | None], None] = lambda *_: None,
) -> list[Output]:
    """Evaluate independent items with bounded admission and ordered results.

    At most ``workers`` items are in flight. Cancellation or failure stops new
    admission and waits for admitted work to exit before returning control.
    The callback also runs inside each worker before and after its kernel.
    """
    selected = options or BTExecutionOptions()
    # Mixed evaluation units cannot nest another numerical executor while a
    # caller owns the process-native permit. Standalone explicit pools remain.
    workers = 1 if selected.native_admission is not None else selected.workers
    limits = selected.limits.for_workers(workers)
    total = len(items) if hasattr(items, "__len__") else None

    def invoke(item: Input) -> Output:
        admission = (selected.native_admission("mixed")
                     if selected.native_admission is not None else nullcontext())
        with resource_limits(limits), admission:
            check_canceled()
            result = evaluate(item)
            check_canceled()
            return result

    results: list[Output] = []
    iterator = iter(items)
    if workers == 1:
        for item in iterator:
            check_canceled()
            results.append(invoke(item))
            progress(len(results), total)
        return results
    ordered: dict[int, Output] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        index = 0

        def admit():
            nonlocal index
            while len(pending) < workers:
                check_canceled()
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending[pool.submit(invoke, item)] = index
                index += 1

        try:
            admit()
            while pending:
                check_canceled()
                done, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                # Observe every already-completed failure before admitting more.
                done.update(future for future in pending if future.done())
                completed = [(future, future.result()) for future in done]
                for future, result in completed:
                    ordered[pending.pop(future)] = result
                    progress(len(ordered), total)
                admit()
        finally:
            for future in pending:
                future.cancel()
    return [ordered[index] for index in range(len(ordered))]
