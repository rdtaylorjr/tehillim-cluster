"""Clustering methods over psalm similarity results."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from tehillim_cluster.k_selection import (
    analyze_k_selection,
    cluster_labels,
    gap_statistic,
    subsample_k_stability,
)
from tehillim_cluster.similarity import SimilarityResult


@dataclass(frozen=True, slots=True)
class KChoice:
    """A chosen cluster count."""

    n_clusters: int
    partition_p_value: float | None
    """`None` for a `fixed_k` choice, since significance doesn't apply to
    a count that wasn't derived from the data at all. Otherwise this
    choice's partition-significance p-value, see
    `k_selection._partition_significance`."""
    stability: float | None
    """`None` for a `fixed_k` choice, for the same reason. Otherwise the
    fraction of subsamples whose silhouette sweep agreed with this
    exact k, see `k_selection.subsample_k_stability`."""


KSelector = Callable[[SimilarityResult], KChoice]
"""A strategy for choosing how many clusters to produce from a given
similarity matrix, see `fixed_k` and `data_driven_k`."""


def fixed_k(n: int) -> KSelector:
    """Always cluster into exactly `n` groups, regardless of the data."""
    return lambda _similarity: KChoice(n_clusters=n, partition_p_value=None, stability=None)


def data_driven_k(
    k_values: Sequence[int],
    *,
    n_subsamples: int = 100,
    n_gap_references: int = 10,
    seed: int = 0,
) -> KSelector:
    """Chooses k via the gap statistic first (returns k=1 if it finds no real structure)."""
    k_values = tuple(k_values)

    def select(similarity: SimilarityResult) -> KChoice:
        """Chooses the cluster count k for this similarity matrix."""
        gap = gap_statistic(
            similarity, k_values=(1, *k_values), n_references=n_gap_references, seed=seed
        )
        if gap.best_k == 1:
            return KChoice(n_clusters=1, partition_p_value=None, stability=None)

        selection = analyze_k_selection(similarity, k_values=k_values)
        chosen_k = selection.best_k_by_silhouette
        stability_result = subsample_k_stability(
            similarity, k_values=k_values, n_subsamples=n_subsamples, seed=seed
        )
        stability = stability_result.win_counts[k_values.index(chosen_k)] / n_subsamples
        return KChoice(
            n_clusters=chosen_k,
            partition_p_value=selection.partition_p_value,
            stability=stability,
        )

    return select


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """One clustering method's partition of the psalms, with the labels it assigned."""

    method: str
    description: str
    psalm_numbers: tuple[int, ...]
    labels: tuple[int, ...]
    """Cluster index per psalm, zero-indexed, same order as `psalm_numbers`."""
    n_clusters: int
    partition_p_value: float | None = None
    """This clustering's k-choice diagnostics, carried through from
    the `KSelector` that produced it, see `KChoice`. `None` for a
    `fixed_k`-configured method, or for a `ClusteringResult` built
    without going through `SpectralClusteringMethod` (e.g. in tests)."""
    k_stability: float | None = None
    """See `KChoice.stability`."""


class ClusteringMethod(Protocol):
    """A named, documented way to turn a SimilarityResult into a psalm partition."""

    @property
    def name(self) -> str:
        """The name this method is reported under."""

    @property
    def description(self) -> str:
        """What the method computes, for whoever reads the output."""

    def compute(self, similarity: SimilarityResult) -> ClusteringResult:
        """Partitions the psalms this similarity matrix covers."""
        ...


@dataclass(frozen=True, slots=True)
class SpectralClusteringMethod:
    """Partitions psalms via spectral clustering over an already-computed similarity matrix."""

    name: str
    description: str
    k_selector: KSelector

    def compute(self, similarity: SimilarityResult) -> ClusteringResult:
        """Partitions the psalms this similarity matrix covers."""
        with warnings.catch_warnings():
            #: Sparse signals can leave isolated nodes in the affinity graph.
            warnings.filterwarnings(
                "ignore", message="Graph is not fully connected", category=UserWarning
            )
            choice = self.k_selector(similarity)
            n_clusters = choice.n_clusters
            # SpectralClustering has no real notion of a single cluster.
            labels = cluster_labels(similarity.matrix, n_clusters)
        return ClusteringResult(
            method=self.name,
            description=self.description,
            psalm_numbers=similarity.psalm_numbers,
            labels=tuple(int(label) for label in labels),
            n_clusters=n_clusters,
            partition_p_value=choice.partition_p_value,
            k_stability=choice.stability,
        )
