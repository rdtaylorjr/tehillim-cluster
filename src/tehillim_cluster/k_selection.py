"""Diagnostics for choosing k from the data: silhouette score."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from tehillim_cluster.parallel import map_in_pool
from tehillim_cluster.similarity import SimilarityResult

#: silhouette is undefined for a single cluster, so two is the smallest k it can score.
_MIN_SILHOUETTE_K = 2

#: One past the largest k any sweep considers, so every sweep stops at the same place.
_MAX_K = 11

#: The k sweep every silhouette-scored method shares.
DEFAULT_K_VALUES = range(_MIN_SILHOUETTE_K, _MAX_K)

#: The gap statistic scores k=1 as its no-cluster null, so its sweep starts one lower.
GAP_K_VALUES = range(1, _MAX_K)

#: Permutations behind every p-value in one payload, so they share a resolution and a floor.
DEFAULT_N_PERMUTATIONS = 2000


@dataclass(frozen=True, slots=True)
class KSelectionResult:
    """Silhouette scores across a range of k, and the k they select."""

    k_values: tuple[int, ...]
    silhouette_scores: tuple[float, ...]
    """Silhouette score of the spectral-clustering partition at each k in
    `k_values`, in the same order. -1.0 (the theoretical minimum) where a
    partition degenerates to fewer than 2 distinct clusters, since
    silhouette is undefined there."""
    eigengaps: tuple[float, ...]
    """Gap between the k-th and (k+1)-th smallest normalized-Laplacian
    eigenvalues, for each k in `k_values`. A large gap suggests k is a
    natural number of clusters for this affinity matrix."""
    best_k_by_silhouette: int
    best_k_by_eigengap: int
    partition_p_value: float
    """Permutation-test p-value for `best_k_by_silhouette`'s partition
    (see `_partition_significance`): the fraction of random same-size
    relabelings whose silhouette score meets or exceeds this partition's.
    Silhouette can never return k=1, so this is the mechanism that flags
    a winning k as indistinguishable from noise."""


def analyze_k_selection(
    similarity: SimilarityResult,
    k_values: Sequence[int] = range(2, 11),
    *,
    partition_permutations: int = 2000,
    seed: int = 0,
) -> KSelectionResult:
    """Silhouette and eigengap scores for every k in `k_values`."""
    k_values = tuple(k_values)
    # Clip to non-negative.
    distance = np.clip(1.0 - similarity.matrix, 0.0, None)
    np.fill_diagonal(distance, 0.0)

    #: Serial on purpose: a pool here measured 5x slower, the win is across methods instead.
    scored = [_silhouette_at_k(similarity.matrix, distance, k) for k in k_values]
    silhouette_scores = tuple(score for score, _ in scored)
    eigenvalues = _normalized_laplacian_eigenvalues(similarity.matrix)
    eigengaps = tuple(float(eigenvalues[k] - eigenvalues[k - 1]) for k in k_values)

    best_k_by_silhouette = k_values[int(np.argmax(silhouette_scores))]
    best_k_by_eigengap = k_values[int(np.argmax(eigengaps))]
    _, winning_labels = scored[k_values.index(best_k_by_silhouette)]
    partition_p_value = _partition_significance(
        distance, winning_labels, n_permutations=partition_permutations, seed=seed
    )

    return KSelectionResult(
        k_values=k_values,
        silhouette_scores=silhouette_scores,
        eigengaps=eigengaps,
        best_k_by_silhouette=best_k_by_silhouette,
        best_k_by_eigengap=best_k_by_eigengap,
        partition_p_value=partition_p_value,
    )


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    """Renumbers clusters by first appearance, so a relabeling cannot read as a new partition."""
    uniques, first_index = np.unique(labels, return_index=True)
    ranked = np.empty(uniques.size, dtype=np.intp)
    ranked[np.argsort(first_index)] = np.arange(uniques.size)
    return ranked[np.searchsorted(uniques, labels)]


def cluster_labels(affinity: np.ndarray, k: int) -> np.ndarray:
    """Spectral-clustering labels for `k` clusters over `affinity`."""
    if k <= 1:
        return np.zeros(affinity.shape[0], dtype=int)
    model = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=0)
    return canonical_labels(np.asarray(model.fit_predict(affinity)))


def _silhouette_at_k(
    affinity: np.ndarray, distance: np.ndarray, k: int
) -> tuple[float, np.ndarray]:
    labels = cluster_labels(affinity, k)
    if len(set(labels)) < 2:
        #: Silhouette is undefined for a single effective cluster, reachable at small k.
        return -1.0, labels
    return float(silhouette_score(distance, labels, metric="precomputed")), labels


def _partition_significance(
    distance: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 2000,
    seed: int = 0,
) -> float:
    """Tests whether `labels`'s silhouette beats a random relabeling of the same distances."""
    if len(set(labels)) < 2:
        #: No real partition to test, which analyze_k_selection reaches via best_k_by_silhouette.
        return 1.0

    observed = float(silhouette_score(distance, labels, metric="precomputed"))
    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_permutations)
    #: Serial on purpose: 2000 of these cost 1s, where spawning a pool for them costs 5s.
    for i in range(n_permutations):
        null_scores[i] = silhouette_score(distance, rng.permutation(labels), metric="precomputed")

    exceedances = int(np.sum(null_scores >= observed))
    return (exceedances + 1) / (n_permutations + 1)


@dataclass(frozen=True, slots=True)
class KStabilityResult:
    """How consistently the silhouette k-sweep picks the same k when the corpus is resampled."""

    k_values: tuple[int, ...]
    win_counts: tuple[int, ...]
    """Number of subsamples (out of `n_subsamples`) whose silhouette
    sweep preferred each k in `k_values`, in the same order."""
    most_stable_k: int
    """Whichever k won the most subsamples."""
    stability: float
    """`most_stable_k`'s win fraction: win_counts[i] / n_subsamples."""
    n_subsamples: int


#: Below this, spawning workers costs more than it saves (measured 0.67x at 25, 1.47x at 100).
_MIN_SUBSAMPLES_FOR_POOL = 50


def _winner_for_sample(k_values: tuple[int, ...], matrix: np.ndarray) -> int:
    """Best k for one subsampled matrix, at module level so a worker process can import it."""
    return _best_k_by_silhouette(matrix, k_values)


def subsample_k_stability(
    similarity: SimilarityResult,
    k_values: Sequence[int] = range(2, 11),
    *,
    n_subsamples: int = 100,
    subsample_fraction: float = 0.8,
    seed: int = 0,
    rng_factory: Callable[[int], Any] = np.random.default_rng,
    max_workers: int | None = None,
    executor_factory: Callable[..., AbstractContextManager[Any]] = ProcessPoolExecutor,
) -> KStabilityResult:
    """Reruns the silhouette k-sweep over random subsamples and tallies which k wins."""
    k_values = tuple(k_values)
    n = len(similarity.psalm_numbers)
    subsample_size = max(2, int(n * subsample_fraction))
    #: silhouette scores 2 <= k <= n-1 labels, and sklearn reports a violation far from its cause.
    for k in k_values:
        if k < _MIN_SILHOUETTE_K:
            raise ValueError(f"silhouette needs at least two clusters, got k={k}")
        if k > subsample_size - 1:
            raise ValueError(
                f"a subsample of {subsample_size} psalms cannot support k={k}: "
                f"silhouette needs at most {subsample_size - 1} clusters"
            )
    rng = rng_factory(seed)

    #: Drawn up front and in order, so the worker count cannot change which samples are scored.
    samples = [rng.choice(n, size=subsample_size, replace=False) for _ in range(n_subsamples)]
    matrices = [similarity.matrix[np.ix_(sample, sample)] for sample in samples]

    winners = map_in_pool(
        partial(_winner_for_sample, k_values),
        matrices,
        max_workers=max_workers,
        minimum_for_pool=_MIN_SUBSAMPLES_FOR_POOL,
        executor_factory=executor_factory,
    )

    win_counts = dict.fromkeys(k_values, 0)
    for winner in winners:
        win_counts[winner] += 1

    most_stable_k = max(k_values, key=lambda k: win_counts[k])
    return KStabilityResult(
        k_values=k_values,
        win_counts=tuple(win_counts[k] for k in k_values),
        most_stable_k=most_stable_k,
        stability=win_counts[most_stable_k] / n_subsamples,
        n_subsamples=n_subsamples,
    )


def _best_k_by_silhouette(similarity_matrix: np.ndarray, k_values: Sequence[int]) -> int:
    distance = np.clip(1.0 - similarity_matrix, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    scores = [_silhouette_at_k(similarity_matrix, distance, k)[0] for k in k_values]
    return k_values[int(np.argmax(scores))]


def _normalized_laplacian_eigenvalues(affinity: np.ndarray) -> np.ndarray:
    """Eigenvalues of the symmetric normalized graph Laplacian `L = I - D^-1/2 A D^-1/2`."""
    degree = affinity.sum(axis=1)
    inv_sqrt_degree = np.diag(1.0 / np.sqrt(degree))
    normalized_affinity = inv_sqrt_degree @ affinity @ inv_sqrt_degree
    laplacian = np.eye(affinity.shape[0]) - normalized_affinity
    return np.linalg.eigvalsh(laplacian)


@dataclass(frozen=True, slots=True)
class GapStatisticResult:
    """The gap statistic (Tibshirani, Walther & Hastie, 2001)."""

    k_values: tuple[int, ...]
    gap: tuple[float, ...]
    """Gap(k), mean log within-cluster dispersion of the reference
    matrices at k minus log dispersion of the real data at k. Larger is
    better-supported."""
    standard_error: tuple[float, ...]
    """Reference-distribution standard error at each k, scaled by
    sqrt(1 + 1/n_references) per Tibshirani et al."""
    best_k: int
    """Smallest k satisfying the standard one-standard-error selection
    rule: Gap(k) >= Gap(k+1) - standard_error(k+1)."""


def gap_statistic(
    similarity: SimilarityResult,
    k_values: Sequence[int] = range(1, 11),
    *,
    n_references: int = 10,
    seed: int = 0,
) -> GapStatisticResult:
    """Gap statistic over `similarity`'s distance matrix (`1 - similarity.matrix`)."""
    k_values = tuple(k_values)
    distance = np.clip(1.0 - similarity.matrix, 0.0, None)
    np.fill_diagonal(distance, 0.0)

    log_wk = np.array(
        [
            _log_within_cluster_dispersion(distance, cluster_labels(similarity.matrix, k))
            for k in k_values
        ]
    )

    rng = np.random.default_rng(seed)
    reference_log_wk = np.empty((n_references, len(k_values)))
    for b in range(n_references):
        reference_distance = _permute_distance_matrix(distance, rng)
        reference_affinity = np.clip(1.0 - reference_distance, 0.0, None)
        np.fill_diagonal(reference_affinity, 1.0)
        for i, k in enumerate(k_values):
            labels = cluster_labels(reference_affinity, k)
            reference_log_wk[b, i] = _log_within_cluster_dispersion(reference_distance, labels)

    mean_reference_log_wk = reference_log_wk.mean(axis=0)
    standard_deviation = reference_log_wk.std(axis=0)
    standard_error = standard_deviation * np.sqrt(1 + 1 / n_references)
    gap = mean_reference_log_wk - log_wk

    best_k = k_values[-1]
    for i in range(len(k_values) - 1):
        if gap[i] >= gap[i + 1] - standard_error[i + 1]:
            best_k = k_values[i]
            break

    return GapStatisticResult(
        k_values=k_values,
        gap=tuple(float(v) for v in gap),
        standard_error=tuple(float(v) for v in standard_error),
        best_k=best_k,
    )


def _log_within_cluster_dispersion(distance: np.ndarray, labels: np.ndarray) -> float:
    """log(Wk), the distance-based within-cluster dispersion, floored before the log."""
    total = 0.0
    for label in set(labels.tolist()):
        members = np.where(labels == label)[0]
        n_r = len(members)
        if n_r < 2:
            continue
        submatrix = distance[np.ix_(members, members)]
        total += float(np.sum(submatrix**2)) / (2 * n_r)
    return float(np.log(max(total, 1e-12)))


def _permute_distance_matrix(distance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A symmetric random permutation of `distance`'s off-diagonal values."""
    n = distance.shape[0]
    upper = np.triu_indices(n, k=1)
    permuted_values = rng.permutation(distance[upper])
    result = np.zeros_like(distance)
    result[upper] = permuted_values
    return np.asarray(result + result.T)
