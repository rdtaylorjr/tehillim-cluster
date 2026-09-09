"""Integration tests for the embedding-based signals the CLI clusters generically."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tehillim_cluster.clustering import ClusteringResult

pytestmark = pytest.mark.integration

ClusteringOf = Callable[[str], ClusteringResult]


def test_every_semantic_clustering_covers_all_150_psalms_with_valid_labels(
    semantic_method: str, clustering_of: ClusteringOf
) -> None:
    """Parametrised over every published non-feature signal, so a new model is covered too."""
    clustering = clustering_of(semantic_method)

    assert len(clustering.psalm_numbers) == 150
    assert len(clustering.labels) == 150
    assert all(0 <= label < clustering.n_clusters for label in clustering.labels)


def test_miqrabert_variants_find_no_real_structure(
    miqrabert_mean_pool_clustering: ClusteringResult,
    miqrabert_soft_alignment_clustering: ClusteringResult,
) -> None:
    assert miqrabert_mean_pool_clustering.n_clusters == 1
    assert miqrabert_soft_alignment_clustering.n_clusters == 1


def test_alephbert_variants_find_real_structure(
    alephbert_mean_pool_clustering: ClusteringResult,
    alephbert_soft_alignment_clustering: ClusteringResult,
) -> None:
    assert alephbert_mean_pool_clustering.n_clusters > 1
    assert alephbert_soft_alignment_clustering.n_clusters > 1
