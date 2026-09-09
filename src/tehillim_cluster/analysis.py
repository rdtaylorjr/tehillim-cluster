"""Group-level statistics over a SimilarityResult."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tehillim_cluster.similarity import SimilarityResult


def mean_pairwise_similarity(result: SimilarityResult, psalm_numbers: Sequence[int]) -> float:
    """Mean similarity across every distinct pair within `psalm_numbers`."""
    if len(psalm_numbers) < 2:
        raise ValueError("need at least two psalms to compute pairwise similarity")

    indices = _indices_for(result, psalm_numbers)
    scores = [
        result.matrix[i, j] for a, i in enumerate(indices) for b, j in enumerate(indices) if a < b
    ]
    return float(np.mean(scores))


def mean_between_group_similarity(
    result: SimilarityResult, group_a: Sequence[int], group_b: Sequence[int]
) -> float:
    """Mean similarity between every psalm in `group_a` and every psalm in `group_b`."""
    if not group_a or not group_b:
        raise ValueError("both groups must be non-empty")

    indices_a = _indices_for(result, group_a)
    indices_b = _indices_for(result, group_b)
    scores = [result.matrix[i, j] for i in indices_a for j in indices_b]
    return float(np.mean(scores))


def _indices_for(result: SimilarityResult, psalm_numbers: Sequence[int]) -> list[int]:
    numbers = list(result.psalm_numbers)
    return [numbers.index(p) for p in psalm_numbers]


@dataclass(frozen=True, slots=True)
class PermutationTestResult:
    """The outcome of a label-permutation significance test."""

    observed: float
    p_value: float
    """One-sided: the fraction of permuted-label draws whose statistic met
    or exceeded the observed one, using Davison & Hinkley's add-one
    smoothing (`(exceedances + 1) / (n_permutations + 1)`). A permutation
    test can never honestly report exactly 0, since the observed grouping
    is itself one arrangement the null could have produced."""
    n_permutations: int
    null_mean: float
    null_std: float


def permutation_test_cohesion(
    result: SimilarityResult,
    group: Sequence[int],
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> PermutationTestResult:
    """Tests whether `group`'s mean pairwise similarity exceeds a random same-size subset's."""
    if len(group) < 2:
        raise ValueError("need at least two psalms to test cohesion")

    index_of = {number: i for i, number in enumerate(result.psalm_numbers)}
    group_indices = np.array([index_of[p] for p in group])
    observed = _mean_similarity_of_indices(result.matrix, group_indices)

    n = len(result.psalm_numbers)
    k = len(group_indices)
    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        sample = rng.choice(n, size=k, replace=False)
        null_scores[i] = _mean_similarity_of_indices(result.matrix, sample)

    return _summarize(observed, null_scores, n_permutations)


def permutation_test_separation(
    result: SimilarityResult,
    group_a: Sequence[int],
    group_b: Sequence[int],
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> PermutationTestResult:
    """Tests whether `group_a`/`group_b` are more distinct than a random resplit."""
    if not group_a or not group_b:
        raise ValueError("both groups must be non-empty")

    index_of = {number: i for i, number in enumerate(result.psalm_numbers)}
    indices_a = np.array([index_of[p] for p in group_a])
    indices_b = np.array([index_of[p] for p in group_b])
    pooled = np.concatenate([indices_a, indices_b])
    size_a = len(indices_a)

    observed = _separation_statistic(result.matrix, indices_a, indices_b)

    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled)
        null_scores[i] = _separation_statistic(result.matrix, shuffled[:size_a], shuffled[size_a:])

    return _summarize(observed, null_scores, n_permutations)


def _mean_similarity_of_indices(matrix: np.ndarray, indices: np.ndarray) -> float:
    """Mean over distinct pairs within `indices`."""
    sub = matrix[np.ix_(indices, indices)]
    n = len(indices)
    return float((sub.sum() - n) / (n * (n - 1)))


def _separation_statistic(
    matrix: np.ndarray, indices_a: np.ndarray, indices_b: np.ndarray
) -> float:
    """Average within-group similarity minus average between-group similarity."""
    within_a = _mean_similarity_of_indices(matrix, indices_a) if len(indices_a) >= 2 else 0.0
    within_b = _mean_similarity_of_indices(matrix, indices_b) if len(indices_b) >= 2 else 0.0
    between = float(matrix[np.ix_(indices_a, indices_b)].mean())
    return 0.5 * (within_a + within_b) - between


def _summarize(
    observed: float, null_scores: np.ndarray, n_permutations: int
) -> PermutationTestResult:
    exceedances = int(np.sum(null_scores >= observed))
    p_value = (exceedances + 1) / (n_permutations + 1)
    return PermutationTestResult(
        observed=float(observed),
        p_value=float(p_value),
        n_permutations=n_permutations,
        null_mean=float(np.mean(null_scores)),
        null_std=float(np.std(null_scores)),
    )


def benjamini_hochberg(p_values: Sequence[float]) -> tuple[float, ...]:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return ()

    order = np.argsort(p)
    ranked = p[order]
    scaled = ranked * n / (np.arange(n) + 1)
    # Step-up: a raw term at rank i can be smaller than one at a higher rank purely from the 1/i.
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    result = np.empty(n)
    result[order] = adjusted_sorted
    return tuple(float(x) for x in result)
