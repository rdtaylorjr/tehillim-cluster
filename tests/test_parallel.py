"""One pool rule: order preserved, no nesting, and no pool when there is nothing to spread."""

from __future__ import annotations

from typing import Self

from tehillim_cluster.parallel import map_in_pool


def _double(value: int) -> int:
    return value * 2


class TestMapInPool:
    def test_preserves_submission_order_so_a_rerun_reports_the_same_sequence(self) -> None:
        assert map_in_pool(_double, [3, 1, 2], max_workers=1) == [6, 2, 4]

    def test_returns_empty_for_no_items_without_starting_a_pool(self) -> None:
        def exploding_factory(**_kwargs: object) -> object:
            raise AssertionError("no pool should start when there is nothing to map")

        assert map_in_pool(_double, [], executor_factory=exploding_factory) == []

    def test_stays_serial_inside_a_worker_so_pools_never_nest(self) -> None:
        """A nested pool would square the process count and exhaust memory."""

        def exploding_factory(**_kwargs: object) -> object:
            raise AssertionError("an inner call must not start its own pool")

        result = map_in_pool(
            _double,
            [1, 2, 3],
            max_workers=4,
            executor_factory=exploding_factory,
            in_worker_process=lambda: True,
        )

        assert result == [2, 4, 6]

    def test_stays_serial_below_the_minimum_worth_spreading(self) -> None:
        def exploding_factory(**_kwargs: object) -> object:
            raise AssertionError("a pool should not start for so few items")

        result = map_in_pool(
            _double,
            [1, 2],
            minimum_for_pool=10,
            executor_factory=exploding_factory,
            in_worker_process=lambda: False,
        )

        assert result == [2, 4]

    def test_an_explicit_worker_count_overrides_the_minimum(self) -> None:
        """An explicit request is a caller's decision, not a heuristic to second-guess."""
        started: list[int] = []

        class _Pool:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def map(self, fn, items, chunksize):
                return [fn(i) for i in items]

        def factory(*, max_workers: int) -> _Pool:
            started.append(max_workers)
            return _Pool()

        map_in_pool(
            _double,
            [1, 2],
            max_workers=2,
            minimum_for_pool=10,
            executor_factory=factory,
            in_worker_process=lambda: False,
        )

        assert started == [2]

    def test_never_starts_more_workers_than_there_are_items(self) -> None:
        started: list[int] = []

        class _Pool:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def map(self, fn, items, chunksize):
                return [fn(i) for i in items]

        def factory(*, max_workers: int) -> _Pool:
            started.append(max_workers)
            return _Pool()

        map_in_pool(
            _double,
            [1, 2],
            max_workers=16,
            executor_factory=factory,
            in_worker_process=lambda: False,
        )

        assert started == [2]
