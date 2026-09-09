"""The k sweep and its permutation null must give the same answer on every run."""

from __future__ import annotations

import numpy as np

from tehillim_cluster.k_selection import analyze_k_selection
from tehillim_cluster.similarity import SimilarityResult

N = 40
K_VALUES = (2, 3, 4)
PERMUTATIONS = 60


def _similarity() -> SimilarityResult:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(N, 8))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    matrix = (raw @ raw.T + 1.0) / 2.0
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    return SimilarityResult(
        method="m", description="d", psalm_numbers=tuple(range(1, N + 1)), matrix=matrix
    )


def _analyze() -> object:
    return analyze_k_selection(_similarity(), K_VALUES, partition_permutations=PERMUTATIONS)


class TestTheAnalysisIsReproducible:
    def test_the_p_value_repeats_exactly(self) -> None:
        """A null drawn from an unseeded source would drift between runs."""
        assert _analyze().partition_p_value == _analyze().partition_p_value

    def test_the_silhouette_scores_repeat_exactly(self) -> None:
        assert _analyze().silhouette_scores == _analyze().silhouette_scores

    def test_the_chosen_k_repeats_exactly(self) -> None:
        assert _analyze().best_k_by_silhouette == _analyze().best_k_by_silhouette


class TestThePermutationPValueKeepsItsForm:
    def test_it_stays_within_the_add_one_bounds(self) -> None:
        """Phipson and Smyth's estimator can never reach zero."""
        result = _analyze()

        assert 1 / (PERMUTATIONS + 1) <= result.partition_p_value <= 1.0

    def test_it_is_a_multiple_of_one_over_n_plus_one(self) -> None:
        result = _analyze()

        assert np.isclose(result.partition_p_value * (PERMUTATIONS + 1) % 1, 0, atol=1e-9)
