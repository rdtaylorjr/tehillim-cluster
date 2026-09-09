"""Cluster ids carry no meaning, so a permutation must not read as a changed partition."""

from __future__ import annotations

import numpy as np

from tehillim_cluster.k_selection import canonical_labels, cluster_labels


class TestCanonicalLabels:
    def test_clusters_are_numbered_by_first_appearance(self) -> None:
        assert list(canonical_labels(np.array([2, 2, 0, 1, 0]))) == [0, 0, 1, 2, 1]

    def test_a_permutation_of_the_same_partition_canonicalises_identically(self) -> None:
        """A rerun would otherwise show this difference for an unchanged partition."""
        first = np.array([0, 0, 1, 1, 2])
        permuted = np.array([2, 2, 0, 0, 1])

        assert list(canonical_labels(first)) == list(canonical_labels(permuted))

    def test_a_genuinely_different_partition_still_differs(self) -> None:
        assert list(canonical_labels(np.array([0, 0, 1, 1]))) != list(
            canonical_labels(np.array([0, 1, 0, 1]))
        )

    def test_already_canonical_labels_are_unchanged(self) -> None:
        labels = np.array([0, 1, 1, 2, 0])

        assert list(canonical_labels(labels)) == list(labels)

    def test_a_single_cluster_is_left_alone(self) -> None:
        assert list(canonical_labels(np.zeros(4, dtype=int))) == [0, 0, 0, 0]

    def test_the_cluster_sizes_are_preserved(self) -> None:
        labels = np.array([3, 3, 1, 7, 7, 7])

        assert sorted(np.bincount(canonical_labels(labels))) == sorted([1, 2, 3])


class TestClusterLabelsAreCanonical:
    def test_the_first_psalm_is_always_in_cluster_zero(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(40, 8))
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        affinity = (raw @ raw.T + 1.0) / 2.0
        np.fill_diagonal(affinity, 1.0)

        assert cluster_labels(affinity, 4)[0] == 0
