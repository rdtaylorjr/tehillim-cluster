from __future__ import annotations

import numpy as np
import pytest

from tehillim_cluster.analysis import (
    benjamini_hochberg,
    mean_between_group_similarity,
    mean_pairwise_similarity,
    permutation_test_cohesion,
    permutation_test_separation,
)
from tehillim_cluster.similarity import SimilarityResult


def _result(matrix: list[list[float]], psalm_numbers: tuple[int, ...]) -> SimilarityResult:
    return SimilarityResult(
        method="test",
        description="test",
        psalm_numbers=psalm_numbers,
        matrix=np.array(matrix),
    )


class TestMeanPairwiseSimilarity:
    def test_averages_every_off_diagonal_pair_within_the_group(self):
        result = _result(
            [
                [1.0, 0.2, 0.4],
                [0.2, 1.0, 0.6],
                [0.4, 0.6, 1.0],
            ],
            psalm_numbers=(1, 2, 3),
        )
        # pairs: (1,2)=0.2, (1,3)=0.4, (2,3)=0.6 -> mean 0.4
        assert mean_pairwise_similarity(result, [1, 2, 3]) == pytest.approx(0.4)

    def test_ignores_psalms_outside_the_requested_group(self):
        result = _result(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.1],
                [0.1, 0.1, 1.0],
            ],
            psalm_numbers=(1, 2, 3),
        )
        assert mean_pairwise_similarity(result, [1, 2]) == pytest.approx(0.9)

    def test_does_not_depend_on_argument_order(self):
        result = _result(
            [[1.0, 0.3], [0.3, 1.0]],
            psalm_numbers=(1, 2),
        )
        assert mean_pairwise_similarity(result, [1, 2]) == mean_pairwise_similarity(result, [2, 1])

    def test_raises_for_a_single_psalm(self):
        result = _result([[1.0]], psalm_numbers=(1,))
        with pytest.raises(ValueError, match="at least two"):
            mean_pairwise_similarity(result, [1])

    def test_raises_for_an_empty_group(self):
        result = _result([[1.0]], psalm_numbers=(1,))
        with pytest.raises(ValueError, match="at least two"):
            mean_pairwise_similarity(result, [])


class TestMeanBetweenGroupSimilarity:
    def test_averages_every_cross_group_pair(self):
        result = _result(
            [
                [1.0, 0.2, 0.4, 0.6],
                [0.2, 1.0, 0.1, 0.3],
                [0.4, 0.1, 1.0, 0.5],
                [0.6, 0.3, 0.5, 1.0],
            ],
            psalm_numbers=(1, 2, 3, 4),
        )
        # group_a={1,2}, group_b={3,4} pairs: (1,3)=0.4 (1,4)=0.6 (2,3)=0.1 (2,4)=0.3 -> mean 0.35
        assert mean_between_group_similarity(result, [1, 2], [3, 4]) == pytest.approx(0.35)

    def test_disjoint_groups_ignore_within_group_pairs(self):
        result = _result(
            [
                [1.0, 0.99, 0.0],
                [0.99, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            psalm_numbers=(1, 2, 3),
        )
        # The 0.99 within {1,2} must not leak into the cross-group average.
        assert mean_between_group_similarity(result, [1, 2], [3]) == pytest.approx(0.0)

    def test_raises_for_an_empty_group(self):
        result = _result([[1.0, 0.5], [0.5, 1.0]], psalm_numbers=(1, 2))
        with pytest.raises(ValueError, match="non-empty"):
            mean_between_group_similarity(result, [], [1])


def _two_block_result(
    within_a: float, within_b: float, between: float, size_a: int = 3, size_b: int = 3
) -> SimilarityResult:
    """A corpus split into two equal-similarity blocks."""
    n = size_a + size_b
    matrix = np.full((n, n), between)
    matrix[:size_a, :size_a] = within_a
    matrix[size_a:, size_a:] = within_b
    np.fill_diagonal(matrix, 1.0)
    return _result(matrix.tolist(), psalm_numbers=tuple(range(1, n + 1)))


class TestPermutationTestCohesion:
    def test_flags_a_group_that_is_far_more_cohesive_than_the_corpus(self):
        # Group {1,2,3} is a tight.
        result = _two_block_result(within_a=0.95, within_b=0.05, between=0.05, size_a=3, size_b=6)
        outcome = permutation_test_cohesion(result, [1, 2, 3], n_permutations=2000, seed=0)
        assert outcome.observed == pytest.approx(0.95)
        # Only the exact subset {1,2,3} matches this cohesion out of the C(9,3)=84 possible.
        assert outcome.p_value < 0.05

    def test_does_not_flag_a_group_from_a_uniform_corpus(self):
        # Every pair has identical similarity.
        result = _two_block_result(within_a=0.4, within_b=0.4, between=0.4, size_a=3, size_b=6)
        outcome = permutation_test_cohesion(result, [1, 2, 3], n_permutations=500, seed=0)
        assert outcome.observed == pytest.approx(0.4)
        assert outcome.p_value == pytest.approx(1.0)

    def test_p_value_is_reproducible_for_a_fixed_seed(self):
        result = _two_block_result(within_a=0.8, within_b=0.2, between=0.3, size_a=3, size_b=6)
        first = permutation_test_cohesion(result, [1, 2, 3], n_permutations=300, seed=42)
        second = permutation_test_cohesion(result, [1, 2, 3], n_permutations=300, seed=42)
        assert first.p_value == second.p_value

    def test_records_the_requested_permutation_count(self):
        result = _two_block_result(within_a=0.8, within_b=0.2, between=0.3, size_a=3, size_b=6)
        outcome = permutation_test_cohesion(result, [1, 2, 3], n_permutations=150, seed=0)
        assert outcome.n_permutations == 150

    def test_raises_for_fewer_than_two_psalms(self):
        result = _result([[1.0]], psalm_numbers=(1,))
        with pytest.raises(ValueError, match="at least two"):
            permutation_test_cohesion(result, [1])


class TestPermutationTestSeparation:
    def test_flags_two_blocks_that_are_far_more_separated_than_chance(self):
        # size 4/4 (pool of 8): only the true split and its mirror complement out of C(8,4)=70.
        result = _two_block_result(within_a=0.9, within_b=0.9, between=0.1, size_a=4, size_b=4)
        outcome = permutation_test_separation(
            result, [1, 2, 3, 4], [5, 6, 7, 8], n_permutations=2000, seed=0
        )
        assert outcome.observed == pytest.approx(0.8)  # 0.5*(0.9+0.9) - 0.1
        assert outcome.p_value < 0.05

    def test_does_not_flag_two_groups_from_a_uniform_corpus(self):
        result = _two_block_result(within_a=0.5, within_b=0.5, between=0.5, size_a=4, size_b=4)
        outcome = permutation_test_separation(
            result, [1, 2, 3, 4], [5, 6, 7, 8], n_permutations=500, seed=0
        )
        assert outcome.observed == pytest.approx(0.0)
        assert outcome.p_value == pytest.approx(1.0)

    def test_p_value_is_reproducible_for_a_fixed_seed(self):
        result = _two_block_result(within_a=0.7, within_b=0.6, between=0.3, size_a=4, size_b=4)
        first = permutation_test_separation(
            result, [1, 2, 3, 4], [5, 6, 7, 8], n_permutations=300, seed=7
        )
        second = permutation_test_separation(
            result, [1, 2, 3, 4], [5, 6, 7, 8], n_permutations=300, seed=7
        )
        assert first.p_value == second.p_value

    def test_raises_for_an_empty_group(self):
        result = _result([[1.0, 0.5], [0.5, 1.0]], psalm_numbers=(1, 2))
        with pytest.raises(ValueError, match="non-empty"):
            permutation_test_separation(result, [], [1])


class TestBenjaminiHochberg:
    def test_matches_a_hand_worked_example(self):
        # p = [0.01, 0.04, 0.03, 0.20], m=4.
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
        assert adjusted[0] == pytest.approx(0.04)
        assert adjusted[1] == pytest.approx(0.04 / 3 * 4)
        assert adjusted[2] == pytest.approx(0.04 / 3 * 4)
        assert adjusted[3] == pytest.approx(0.20)

    def test_identical_p_values_are_unchanged(self):
        # p_i * m / i is exactly 0.05 for every i when p = [0.01, 0.02, 0.03, 0.04.
        adjusted = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
        for value in adjusted:
            assert value == pytest.approx(0.05)

    def test_every_adjusted_value_is_at_least_the_raw_p_value(self):
        # A defining property of the BH procedure: correction can only make a p-value larger.
        raw = [0.001, 0.03, 0.055, 0.2, 0.4, 0.9]
        adjusted = benjamini_hochberg(raw)
        for r, a in zip(raw, adjusted, strict=True):
            assert a >= r - 1e-12

    def test_adjusted_values_are_clipped_to_one(self):
        adjusted = benjamini_hochberg([0.9, 0.95, 0.99, 1.0])
        assert all(value <= 1.0 for value in adjusted)

    def test_preserves_input_order_and_length(self):
        adjusted = benjamini_hochberg([0.5, 0.01, 0.3])
        assert len(adjusted) == 3
        # The smallest raw p-value should still adjust to the smallest (or tied-smallest) result.
        assert adjusted[1] <= adjusted[0]
        assert adjusted[1] <= adjusted[2]

    def test_empty_input_returns_empty_tuple(self):
        assert benjamini_hochberg([]) == ()
