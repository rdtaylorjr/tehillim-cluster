from __future__ import annotations

import numpy as np
import pytest

from tehillim_cluster.k_selection import analyze_k_selection, gap_statistic, subsample_k_stability
from tehillim_cluster.similarity import SimilarityResult


def _block_similarity(
    n_blocks: int, block_size: int, within: float, between: float
) -> SimilarityResult:
    n = n_blocks * block_size
    matrix = np.full((n, n), between)
    for start in range(0, n, block_size):
        matrix[start : start + block_size, start : start + block_size] = within
    np.fill_diagonal(matrix, 1.0)
    return SimilarityResult(
        method="test", description="test", psalm_numbers=tuple(range(1, n + 1)), matrix=matrix
    )


def _uniform_similarity(n: int, value: float) -> SimilarityResult:
    matrix = np.full((n, n), value)
    np.fill_diagonal(matrix, 1.0)
    return SimilarityResult(
        method="test", description="test", psalm_numbers=tuple(range(1, n + 1)), matrix=matrix
    )


class TestAnalyzeKSelection:
    def test_recovers_the_true_number_of_well_separated_blocks_by_silhouette(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        result = analyze_k_selection(similarity, k_values=range(2, 7))
        assert result.best_k_by_silhouette == 3

    def test_recovers_the_true_number_of_well_separated_blocks_by_eigengap(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        result = analyze_k_selection(similarity, k_values=range(2, 7))
        assert result.best_k_by_eigengap == 3

    def test_result_arrays_align_with_the_requested_k_values(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        k_values = tuple(range(2, 8))
        result = analyze_k_selection(similarity, k_values=k_values)
        assert result.k_values == k_values
        assert len(result.silhouette_scores) == len(k_values)
        assert len(result.eigengaps) == len(k_values)

    def test_silhouette_scores_are_within_valid_bounds(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        result = analyze_k_selection(similarity, k_values=range(2, 7))
        assert all(-1.0 <= score <= 1.0 for score in result.silhouette_scores)

    def test_a_uniform_corpus_has_a_far_weaker_eigengap_signal_than_real_blocks(self):
        # No real cluster structure exists.
        blocked = analyze_k_selection(
            _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05),
            k_values=range(2, 7),
        )
        uniform = analyze_k_selection(_uniform_similarity(12, 0.4), k_values=range(2, 7))
        assert max(uniform.eigengaps) < 0.1 * max(blocked.eigengaps)

    def test_best_k_values_are_drawn_from_the_requested_range(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        k_values = range(2, 7)
        result = analyze_k_selection(similarity, k_values=k_values)
        assert result.best_k_by_silhouette in k_values
        assert result.best_k_by_eigengap in k_values

    def test_tolerates_a_similarity_value_that_rounds_slightly_above_one(self):
        """Cosine of near-identical psalms can land a float step above 1.0, which is fine."""
        k_values = range(2, 5)
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        similarity.matrix[0, 1] = similarity.matrix[1, 0] = 1.0 + 1e-10

        result = analyze_k_selection(similarity, k_values=k_values)

        assert result.best_k_by_silhouette in k_values
        assert np.all(np.isfinite(result.silhouette_scores))

    def test_deterministic_across_repeated_calls(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        first = analyze_k_selection(similarity, k_values=range(2, 7))
        second = analyze_k_selection(similarity, k_values=range(2, 7))
        assert first == second


class TestPartitionSignificance:
    def test_flags_a_well_separated_partition_as_significant(self):
        # Three tight.
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.95, between=0.05)
        result = analyze_k_selection(similarity, k_values=range(2, 7), partition_permutations=1000)
        assert result.best_k_by_silhouette == 3
        assert result.partition_p_value < 0.05

    def test_does_not_flag_a_uniform_corpus_as_significant(self):
        # No real structure at all - whichever k silhouette happens to pick.
        similarity = _uniform_similarity(12, 0.4)
        result = analyze_k_selection(similarity, k_values=range(2, 7), partition_permutations=500)
        assert result.partition_p_value == pytest.approx(1.0)

    def test_p_value_is_within_valid_bounds(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        result = analyze_k_selection(similarity, k_values=range(2, 7), partition_permutations=200)
        assert 0.0 < result.partition_p_value <= 1.0

    def test_reproducible_for_a_fixed_seed(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        first = analyze_k_selection(
            similarity, k_values=range(2, 7), partition_permutations=200, seed=7
        )
        second = analyze_k_selection(
            similarity, k_values=range(2, 7), partition_permutations=200, seed=7
        )
        assert first.partition_p_value == second.partition_p_value


class TestSubsampleKStability:
    def test_reports_high_stability_for_strongly_separated_blocks(self):
        similarity = _block_similarity(n_blocks=3, block_size=5, within=0.95, between=0.05)
        result = subsample_k_stability(similarity, k_values=range(2, 7), n_subsamples=50, seed=0)
        assert result.most_stable_k == 3
        assert result.stability > 0.8

    def test_win_counts_sum_to_n_subsamples(self):
        similarity = _block_similarity(n_blocks=3, block_size=5, within=0.9, between=0.05)
        result = subsample_k_stability(similarity, k_values=range(2, 7), n_subsamples=40, seed=0)
        assert sum(result.win_counts) == 40
        assert result.n_subsamples == 40

    def test_win_counts_align_with_k_values(self):
        similarity = _block_similarity(n_blocks=3, block_size=5, within=0.9, between=0.05)
        k_values = tuple(range(2, 7))
        result = subsample_k_stability(similarity, k_values=k_values, n_subsamples=30, seed=0)
        assert result.k_values == k_values
        assert len(result.win_counts) == len(k_values)

    def test_stability_matches_the_most_stable_k_win_fraction(self):
        similarity = _block_similarity(n_blocks=3, block_size=5, within=0.9, between=0.05)
        result = subsample_k_stability(similarity, k_values=range(2, 7), n_subsamples=30, seed=0)
        winner_index = result.k_values.index(result.most_stable_k)
        assert result.stability == pytest.approx(result.win_counts[winner_index] / 30)

    def test_most_stable_k_is_drawn_from_the_requested_range(self):
        similarity = _block_similarity(n_blocks=3, block_size=5, within=0.9, between=0.05)
        k_values = range(2, 7)
        result = subsample_k_stability(similarity, k_values=k_values, n_subsamples=30, seed=0)
        assert result.most_stable_k in k_values

    def test_reproducible_for_a_fixed_seed(self):
        similarity = _block_similarity(n_blocks=3, block_size=5, within=0.9, between=0.05)
        first = subsample_k_stability(similarity, k_values=range(2, 7), n_subsamples=30, seed=3)
        second = subsample_k_stability(similarity, k_values=range(2, 7), n_subsamples=30, seed=3)
        assert first == second

    def test_never_draws_the_same_psalm_twice_in_one_subsample(self):
        #: A duplicated index would make a psalm trivially self-similar with itself.
        similarity = _block_similarity(n_blocks=3, block_size=5, within=0.9, between=0.05)
        seen_samples: list[np.ndarray] = []
        real_rng = np.random.default_rng(0)

        class RecordingRNG:
            """Records every draw while delegating to a real generator."""

            def choice(self, *args, **kwargs):
                """Delegates to the real generator and records the result."""
                result = real_rng.choice(*args, **kwargs)
                seen_samples.append(result)
                return result

        subsample_k_stability(
            similarity,
            k_values=range(2, 5),
            n_subsamples=10,
            seed=0,
            rng_factory=lambda _seed: RecordingRNG(),
        )

        assert seen_samples, "expected subsample_k_stability to draw without replacement"
        for sample in seen_samples:
            assert len(set(sample.tolist())) == len(sample)

    def test_respects_a_custom_subsample_fraction(self):
        similarity = _block_similarity(n_blocks=3, block_size=5, within=0.9, between=0.05)
        result = subsample_k_stability(
            similarity, k_values=range(2, 7), n_subsamples=20, subsample_fraction=0.5, seed=0
        )
        assert sum(result.win_counts) == 20


class TestGapStatistic:
    def test_does_not_select_k1_when_real_block_structure_exists(self):
        # Three tight.
        similarity = _block_similarity(n_blocks=3, block_size=5, within=0.95, between=0.05)
        result = gap_statistic(similarity, k_values=range(1, 7), n_references=10, seed=0)
        assert result.best_k > 1

    def test_selects_k1_for_a_corpus_with_no_real_structure(self):
        # The regression case this whole diagnostic exists for: silhouette can never say "no real.
        similarity = _uniform_similarity(12, 0.4)
        result = gap_statistic(similarity, k_values=range(1, 7), n_references=10, seed=0)
        assert result.best_k == 1

    def test_k_values_and_gap_arrays_align(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        k_values = tuple(range(1, 6))
        result = gap_statistic(similarity, k_values=k_values, n_references=5, seed=0)
        assert result.k_values == k_values
        assert len(result.gap) == len(k_values)
        assert len(result.standard_error) == len(k_values)

    def test_best_k_is_drawn_from_the_requested_range(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        k_values = range(1, 6)
        result = gap_statistic(similarity, k_values=k_values, n_references=5, seed=0)
        assert result.best_k in k_values

    def test_reproducible_for_a_fixed_seed(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        first = gap_statistic(similarity, k_values=range(1, 6), n_references=5, seed=7)
        second = gap_statistic(similarity, k_values=range(1, 6), n_references=5, seed=7)
        assert first == second

    def test_standard_errors_are_non_negative(self):
        similarity = _block_similarity(n_blocks=3, block_size=4, within=0.9, between=0.05)
        result = gap_statistic(similarity, k_values=range(1, 6), n_references=5, seed=0)
        assert all(se >= 0.0 for se in result.standard_error)


class TestSubsampleStabilityParallelism:
    def test_parallel_and_serial_execution_agree_exactly(self):
        """Samples are drawn up front, so worker count cannot change which k wins."""
        similarity = _block_similarity(n_blocks=3, block_size=5, within=0.9, between=0.05)

        serial = subsample_k_stability(
            similarity, k_values=range(2, 6), n_subsamples=12, seed=5, max_workers=1
        )
        parallel = subsample_k_stability(
            similarity, k_values=range(2, 6), n_subsamples=12, seed=5, max_workers=4
        )

        assert serial == parallel

    def test_a_single_worker_never_starts_a_pool(self):
        """The pool costs more than it saves for a handful of subsamples."""
        started: list[int] = []

        class RecordingExecutor:
            def __init__(self, max_workers=None):
                started.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def map(self, fn, items, chunksize=1):
                return [fn(item) for item in items]

        similarity = _block_similarity(n_blocks=2, block_size=4, within=0.9, between=0.05)
        subsample_k_stability(
            similarity,
            k_values=range(2, 4),
            n_subsamples=6,
            seed=0,
            max_workers=1,
            executor_factory=RecordingExecutor,
        )

        assert started == []


class TestACorpusTooSmallForTheRequestedK:
    """silhouette needs 2 <= k <= n-1, and sklearn reports a violation far from its cause."""

    @staticmethod
    def _similarity(n: int) -> SimilarityResult:
        """Two similar blocks, so the similarity graph is connected and spectral has structure."""
        rng = np.random.default_rng(0)
        block = rng.uniform(0.6, 0.9, size=(n, n))
        matrix = (block + block.T) / 2
        matrix[: n // 2, n // 2 :] = 0.2
        matrix[n // 2 :, : n // 2] = 0.2
        np.fill_diagonal(matrix, 1.0)
        return SimilarityResult(
            method="m",
            description="d",
            psalm_numbers=tuple(range(1, n + 1)),
            matrix=matrix,
        )

    def test_names_the_subsample_size_that_cannot_support_the_largest_k(self) -> None:
        with pytest.raises(ValueError, match="subsample of 3 psalms cannot support k=3"):
            subsample_k_stability(
                self._similarity(4), k_values=(2, 3), n_subsamples=4, max_workers=1
            )

    def test_accepts_a_k_the_subsample_can_support(self) -> None:
        result = subsample_k_stability(
            self._similarity(10), k_values=(2, 3), n_subsamples=4, max_workers=1
        )

        assert result.most_stable_k in (2, 3)

    def test_rejects_a_k_below_two_which_silhouette_cannot_score(self) -> None:
        with pytest.raises(ValueError, match="k=1"):
            subsample_k_stability(
                self._similarity(20), k_values=(1, 2), n_subsamples=4, max_workers=1
            )
