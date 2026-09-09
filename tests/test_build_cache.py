from __future__ import annotations

import numpy as np

from tehillim_cluster.build_cache import (
    clustering_cache_path,
    load_cached_clustering,
    load_cached_similarity,
    similarity_cache_path,
    write_cached_clustering,
    write_cached_similarity,
)
from tehillim_cluster.clustering import ClusteringResult
from tehillim_cluster.similarity import SimilarityResult

FINGERPRINT = "fingerprint-of-the-inputs"


def _clustering_result(**overrides) -> ClusteringResult:
    defaults = {
        "method": "miqrabert-mean-pool-spectral",
        "description": "A test clustering result.",
        "psalm_numbers": (1, 2, 3),
        "labels": (0, 1, 0),
        "n_clusters": 2,
        "partition_p_value": 0.01,
        "k_stability": 0.9,
    }
    defaults.update(overrides)
    return ClusteringResult(**defaults)


def _similarity_result(**overrides) -> SimilarityResult:
    defaults = {
        "method": "miqrabert-mean-pool-cosine",
        "description": "A test similarity result.",
        "psalm_numbers": (1, 2, 3),
        "matrix": np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype="<f4"),
    }
    defaults.update(overrides)
    return SimilarityResult(**defaults)


class TestClusteringCachePath:
    def test_names_the_file_after_the_method(self, tmp_path):
        assert clustering_cache_path(tmp_path, "alephbert-mean-pool-spectral") == (
            tmp_path / "alephbert-mean-pool-spectral.clustering.json"
        )


class TestLoadCachedClustering:
    def test_returns_none_when_no_cache_file_exists(self, tmp_path):
        assert (
            load_cached_clustering(
                tmp_path, "miqrabert-mean-pool-spectral", fingerprint=FINGERPRINT
            )
            is None
        )

    def test_round_trips_every_field(self, tmp_path):
        result = _clustering_result()
        write_cached_clustering(tmp_path, result, fingerprint=FINGERPRINT)

        loaded = load_cached_clustering(tmp_path, result.method, fingerprint=FINGERPRINT)

        assert loaded == result

    def test_round_trips_a_none_partition_p_value_and_stability(self, tmp_path):
        result = _clustering_result(partition_p_value=None, k_stability=None, n_clusters=1)
        write_cached_clustering(tmp_path, result, fingerprint=FINGERPRINT)

        loaded = load_cached_clustering(tmp_path, result.method, fingerprint=FINGERPRINT)

        assert loaded == result

    def test_only_finds_the_matching_method_name(self, tmp_path):
        write_cached_clustering(
            tmp_path,
            _clustering_result(method="miqrabert-mean-pool-spectral"),
            fingerprint=FINGERPRINT,
        )

        assert (
            load_cached_clustering(
                tmp_path, "alephbert-mean-pool-spectral", fingerprint=FINGERPRINT
            )
            is None
        )


class TestWriteCachedClustering:
    def test_creates_the_cache_dir_if_missing(self, tmp_path):
        cache_dir = tmp_path / "nested" / "cache"
        write_cached_clustering(cache_dir, _clustering_result(), fingerprint=FINGERPRINT)
        assert clustering_cache_path(cache_dir, "miqrabert-mean-pool-spectral").exists()

    def test_overwrites_an_existing_cache_file_for_the_same_method(self, tmp_path):
        write_cached_clustering(tmp_path, _clustering_result(n_clusters=2), fingerprint=FINGERPRINT)
        write_cached_clustering(tmp_path, _clustering_result(n_clusters=5), fingerprint=FINGERPRINT)

        loaded = load_cached_clustering(
            tmp_path, "miqrabert-mean-pool-spectral", fingerprint=FINGERPRINT
        )

        assert loaded.n_clusters == 5


class TestSimilarityCachePath:
    def test_names_the_file_after_the_method(self, tmp_path):
        assert similarity_cache_path(tmp_path, "alephbert-mean-pool-cosine") == (
            tmp_path / "alephbert-mean-pool-cosine.similarity.json"
        )

    def test_does_not_collide_with_a_clustering_cache_file_of_the_same_stem(self, tmp_path):
        assert clustering_cache_path(tmp_path, "x") != similarity_cache_path(tmp_path, "x")


class TestLoadCachedSimilarity:
    def test_returns_none_when_no_cache_file_exists(self, tmp_path):
        assert (
            load_cached_similarity(tmp_path, "miqrabert-mean-pool-cosine", fingerprint=FINGERPRINT)
            is None
        )

    def test_round_trips_every_field(self, tmp_path):
        result = _similarity_result()
        write_cached_similarity(tmp_path, result, fingerprint=FINGERPRINT)

        loaded = load_cached_similarity(tmp_path, result.method, fingerprint=FINGERPRINT)

        assert loaded.method == result.method
        assert loaded.description == result.description
        assert loaded.psalm_numbers == result.psalm_numbers
        assert np.array_equal(loaded.matrix, result.matrix)

    def test_round_trips_the_matrix_byte_exact(self, tmp_path):
        """It stored float32, so a cache hit returned a lower-precision matrix than it wrote."""
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((5, 5))
        result = _similarity_result(matrix=matrix, psalm_numbers=(1, 2, 3, 4, 5))
        write_cached_similarity(tmp_path, result, fingerprint=FINGERPRINT)

        loaded = load_cached_similarity(tmp_path, result.method, fingerprint=FINGERPRINT)

        assert np.array_equal(loaded.matrix, matrix)
        assert loaded.matrix.dtype == np.dtype("<f8")

    def test_only_finds_the_matching_method_name(self, tmp_path):
        write_cached_similarity(
            tmp_path,
            _similarity_result(method="miqrabert-mean-pool-cosine"),
            fingerprint=FINGERPRINT,
        )

        assert (
            load_cached_similarity(tmp_path, "alephbert-mean-pool-cosine", fingerprint=FINGERPRINT)
            is None
        )


class TestWriteCachedSimilarity:
    def test_creates_the_cache_dir_if_missing(self, tmp_path):
        cache_dir = tmp_path / "nested" / "cache"
        write_cached_similarity(cache_dir, _similarity_result(), fingerprint=FINGERPRINT)
        assert similarity_cache_path(cache_dir, "miqrabert-mean-pool-cosine").exists()

    def test_overwrites_an_existing_cache_file_for_the_same_method(self, tmp_path):
        first = np.zeros((2, 2), dtype="<f4")
        second = np.ones((2, 2), dtype="<f4")
        write_cached_similarity(
            tmp_path,
            _similarity_result(matrix=first, psalm_numbers=(1, 2)),
            fingerprint=FINGERPRINT,
        )
        write_cached_similarity(
            tmp_path,
            _similarity_result(matrix=second, psalm_numbers=(1, 2)),
            fingerprint=FINGERPRINT,
        )

        loaded = load_cached_similarity(
            tmp_path, "miqrabert-mean-pool-cosine", fingerprint=FINGERPRINT
        )

        assert np.array_equal(loaded.matrix, second)
