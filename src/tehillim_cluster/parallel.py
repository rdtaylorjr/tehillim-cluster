"""One rule for running independent work across processes, shared by every batch in this package."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def in_worker_process() -> bool:
    """True when this process is itself a pool worker, whose parent already claimed the cores."""
    return multiprocessing.parent_process() is not None


def map_in_pool[ItemT, ResultT](
    worker: Callable[[ItemT], ResultT],
    items: Sequence[ItemT],
    *,
    max_workers: int | None = None,
    minimum_for_pool: int = 1,
    executor_factory: Callable[..., AbstractContextManager[Any]] = ProcessPoolExecutor,
    in_worker_process: Callable[[], bool] = in_worker_process,
) -> list[ResultT]:
    """Runs `worker` over every item across processes, results in submission order."""
    #: `worker` must be module-level: the pool pickles it by qualified name to reach each process.
    if not items:
        return []
    requested = max_workers if max_workers is not None else (os.cpu_count() or 1)
    workers = max(1, min(requested, len(items)))
    #: Below this many items a pool costs more to start than the work it would spread.
    if max_workers is None and len(items) < minimum_for_pool:
        workers = 1
    #: Nesting would square the process count and exhaust memory, so an inner call stays serial.
    if workers == 1 or in_worker_process():
        return [worker(item) for item in items]
    with executor_factory(max_workers=workers) as pool:
        return list(pool.map(worker, items, chunksize=-(-len(items) // workers)))
