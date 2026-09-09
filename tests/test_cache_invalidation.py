"""A cache keyed by method name alone returns a prior run's numbers for a new run's inputs."""

from __future__ import annotations

import numpy as np

from tehillim_cluster.build_cache import (
    fingerprint_embeddings,
    input_fingerprint,
    load_cached_clustering,
    load_cached_similarity,
    write_cached_clustering,
    write_cached_similarity,
)
from tehillim_cluster.clustering import ClusteringResult
from tehillim_cluster.similarity import SimilarityResult

METHOD = "miqrabert-mean-pool-cosine"


def _similarity(matrix: np.ndarray) -> SimilarityResult:
    return SimilarityResult(method=METHOD, description="d", psalm_numbers=(1, 2), matrix=matrix)


def _clustering(labels: tuple[int, ...] = (0, 1)) -> ClusteringResult:
    return ClusteringResult(
        method="m",
        description="d",
        psalm_numbers=(1, 2),
        labels=labels,
        n_clusters=2,
        partition_p_value=0.01,
        k_stability=0.9,
    )


class TestSimilarityCacheInvalidation:
    def test_a_hit_needs_the_fingerprint_the_entry_was_written_with(self, tmp_path) -> None:
        """Reruns against new embeddings silently returned the previous run's matrix."""
        write_cached_similarity(tmp_path, _similarity(np.eye(2)), fingerprint="old-data")

        assert load_cached_similarity(tmp_path, METHOD, fingerprint="new-data") is None

    def test_a_matching_fingerprint_still_hits(self, tmp_path) -> None:
        write_cached_similarity(tmp_path, _similarity(np.eye(2)), fingerprint="same")

        loaded = load_cached_similarity(tmp_path, METHOD, fingerprint="same")

        assert loaded is not None
        assert loaded.method == METHOD

    def test_an_entry_written_before_fingerprinting_is_a_miss(self, tmp_path) -> None:
        """Its matrix bytes are float32, so decoding one as float64 would return noise."""
        write_cached_similarity(tmp_path, _similarity(np.eye(2)), fingerprint="x")
        path = tmp_path / f"{METHOD}.similarity.json"
        path.write_text(
            path.read_text(encoding="utf-8").replace('"fingerprint": "x"', '"unused": "x"'),
            encoding="utf-8",
        )

        assert load_cached_similarity(tmp_path, METHOD, fingerprint="x") is None


class TestSimilarityCachePrecision:
    def test_the_cache_round_trip_preserves_every_bit(self, tmp_path) -> None:
        """It stored float32, so a cached read silently undid the float64 computation."""
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(2, 2))
        matrix = (matrix + matrix.T) / 2

        write_cached_similarity(tmp_path, _similarity(matrix), fingerprint="f")
        loaded = load_cached_similarity(tmp_path, METHOD, fingerprint="f")

        assert loaded is not None
        assert loaded.matrix.dtype == np.float64
        assert np.array_equal(loaded.matrix, matrix)


class TestClusteringCacheInvalidation:
    def test_a_hit_needs_the_fingerprint_the_entry_was_written_with(self, tmp_path) -> None:
        write_cached_clustering(tmp_path, _clustering(), fingerprint="old-similarity")

        assert load_cached_clustering(tmp_path, "m", fingerprint="new-similarity") is None

    def test_a_matching_fingerprint_still_hits(self, tmp_path) -> None:
        write_cached_clustering(tmp_path, _clustering(), fingerprint="same")

        assert load_cached_clustering(tmp_path, "m", fingerprint="same") is not None


class TestFingerprints:
    def test_different_embeddings_fingerprint_differently(self) -> None:
        first = {1: np.zeros((2, 3), dtype=np.float32)}
        second = {1: np.ones((2, 3), dtype=np.float32)}

        assert fingerprint_embeddings(first) != fingerprint_embeddings(second)

    def test_the_same_embeddings_fingerprint_identically(self) -> None:
        of = lambda: {2: np.arange(6, dtype=np.float32).reshape(2, 3)}  # noqa: E731

        assert fingerprint_embeddings(of()) == fingerprint_embeddings(of())

    def test_psalm_numbers_are_part_of_the_fingerprint(self) -> None:
        """Two corpora can hold identical vectors under different psalm numbers."""
        rows = np.arange(6, dtype=np.float32).reshape(2, 3)

        assert fingerprint_embeddings({1: rows}) != fingerprint_embeddings({2: rows})

    def test_dtype_is_part_of_the_fingerprint(self) -> None:
        rows = np.arange(6).reshape(2, 3)

        assert input_fingerprint(rows.astype(np.float32)) != input_fingerprint(
            rows.astype(np.float64)
        )

    def test_shape_is_part_of_the_fingerprint(self) -> None:
        rows = np.arange(6, dtype=np.float32)

        assert input_fingerprint(rows.reshape(2, 3)) != input_fingerprint(rows.reshape(3, 2))
