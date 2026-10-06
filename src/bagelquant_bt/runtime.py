"""Explicit bounded local execution; global admission remains with callers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
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

    def __post_init__(self) -> None:
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
    limits = selected.limits.for_workers(selected.workers)
    total = len(items) if hasattr(items, "__len__") else None

    def invoke(item: Input) -> Output:
        with resource_limits(limits):
            check_canceled()
            result = evaluate(item)
            check_canceled()
            return result

    results: list[Output] = []
    iterator = iter(items)
    if selected.workers == 1:
        for item in iterator:
            check_canceled()
            results.append(invoke(item))
            progress(len(results), total)
        return results
    with ThreadPoolExecutor(max_workers=selected.workers) as pool:
        pending = []
        try:
            for _ in range(selected.workers):
                check_canceled()
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.append(pool.submit(invoke, item))
            while pending:
                check_canceled()
                results.append(pending.pop(0).result())
                progress(len(results), total)
                check_canceled()
                for future in pending:
                    if future.done():
                        future.result()
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                pending.append(pool.submit(invoke, item))
        finally:
            for future in pending:
                future.cancel()
    return results
