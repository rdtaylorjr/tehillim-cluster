from __future__ import annotations

import numpy as np

from tehillim_cluster.clustering import SpectralClusteringMethod, data_driven_k, fixed_k
from tehillim_cluster.similarity import SimilarityResult


def _uniform_similarity(psalm_numbers: tuple[int, ...], value: float) -> SimilarityResult:
    n = len(psalm_numbers)
    matrix = np.full((n, n), value)
    np.fill_diagonal(matrix, 1.0)
    return SimilarityResult(
        method="test-similarity", description="test", psalm_numbers=psalm_numbers, matrix=matrix
    )


def _block_similarity(psalm_numbers: tuple[int, ...], block_size: int) -> SimilarityResult:
    """A similarity matrix with well-separated blocks scoring 0.9 within each block."""
    n = len(psalm_numbers)
    matrix = np.full((n, n), 0.05)
    for start in range(0, n, block_size):
        matrix[start : start + block_size, start : start + block_size] = 0.9
    np.fill_diagonal(matrix, 1.0)
    return SimilarityResult(
        method="test-similarity", description="test", psalm_numbers=psalm_numbers, matrix=matrix
    )


_METHOD = SpectralClusteringMethod(
    name="test-cluster", description="test cluster desc", k_selector=fixed_k(2)
)


def test_recovers_two_well_separated_blocks():
    psalm_numbers = tuple(range(1, 9))  # 8 psalms, two blocks of 4
    similarity = _block_similarity(psalm_numbers, block_size=4)
    result = _METHOD.compute(similarity)
    first_block_labels = set(result.labels[:4])
    second_block_labels = set(result.labels[4:])
    assert len(first_block_labels) == 1
    assert len(second_block_labels) == 1
    assert first_block_labels != second_block_labels


def test_recovers_three_well_separated_blocks():
    psalm_numbers = tuple(range(1, 10))  # 9 psalms, three blocks of 3
    similarity = _block_similarity(psalm_numbers, block_size=3)
    method = SpectralClusteringMethod(name="three-block", description="test", k_selector=fixed_k(3))
    result = method.compute(similarity)
    block_labels = [set(result.labels[0:3]), set(result.labels[3:6]), set(result.labels[6:9])]
    for labels in block_labels:
        assert len(labels) == 1
    assert block_labels[0] != block_labels[1]
    assert block_labels[1] != block_labels[2]
    assert block_labels[0] != block_labels[2]


def test_result_preserves_psalm_numbers():
    psalm_numbers = tuple(range(1, 9))
    similarity = _block_similarity(psalm_numbers, block_size=4)
    result = _METHOD.compute(similarity)
    assert result.psalm_numbers == psalm_numbers


def test_result_has_one_label_per_psalm():
    psalm_numbers = tuple(range(1, 9))
    similarity = _block_similarity(psalm_numbers, block_size=4)
    result = _METHOD.compute(similarity)
    assert len(result.labels) == len(psalm_numbers)


def test_result_carries_the_method_name_and_description_it_was_configured_with():
    method = SpectralClusteringMethod(
        name="my-cluster-method", description="my description", k_selector=fixed_k(2)
    )
    psalm_numbers = tuple(range(1, 9))
    similarity = _block_similarity(psalm_numbers, block_size=4)
    result = method.compute(similarity)
    assert result.method == "my-cluster-method"
    assert result.description == "my description"


def test_result_records_the_configured_n_clusters():
    psalm_numbers = tuple(range(1, 9))
    similarity = _block_similarity(psalm_numbers, block_size=4)
    result = _METHOD.compute(similarity)
    assert result.n_clusters == 2


def test_labels_are_zero_indexed_and_within_range():
    psalm_numbers = tuple(range(1, 9))
    similarity = _block_similarity(psalm_numbers, block_size=4)
    result = _METHOD.compute(similarity)
    assert all(0 <= label < result.n_clusters for label in result.labels)


def test_is_deterministic_given_the_same_input():
    psalm_numbers = tuple(range(1, 9))
    similarity = _block_similarity(psalm_numbers, block_size=4)
    result_a = _METHOD.compute(similarity)
    result_b = _METHOD.compute(similarity)
    assert result_a.labels == result_b.labels


def test_two_configured_instances_are_independent():
    # A regression guard against accidentally sharing mutable state between differently-configured.
    method_a = SpectralClusteringMethod(name="a", description="a desc", k_selector=fixed_k(2))
    method_b = SpectralClusteringMethod(name="b", description="b desc", k_selector=fixed_k(3))
    psalm_numbers = tuple(range(1, 9))
    similarity = _block_similarity(psalm_numbers, block_size=4)

    result_a = method_a.compute(similarity)
    result_b = method_b.compute(similarity)

    assert result_a.method == "a"
    assert result_a.n_clusters == 2
    assert result_b.method == "b"
    assert result_b.n_clusters == 3


class TestFixedK:
    def test_always_returns_the_configured_count_regardless_of_data(self):
        selector = fixed_k(4)
        similarity = _block_similarity(tuple(range(1, 9)), block_size=4)
        assert selector(similarity).n_clusters == 4
        assert selector(similarity).n_clusters == 4  # calling twice: no hidden state

    def test_reports_no_diagnostics_since_the_choice_is_not_data_driven(self):
        selector = fixed_k(4)
        similarity = _block_similarity(tuple(range(1, 9)), block_size=4)
        choice = selector(similarity)
        assert choice.partition_p_value is None
        assert choice.stability is None

    def test_a_method_built_from_it_records_no_diagnostics_on_the_result(self):
        method = SpectralClusteringMethod(name="fixed", description="test", k_selector=fixed_k(2))
        similarity = _block_similarity(tuple(range(1, 9)), block_size=4)
        result = method.compute(similarity)
        assert result.partition_p_value is None
        assert result.k_stability is None


class TestDataDrivenK:
    def test_picks_the_true_number_of_well_separated_blocks(self):
        selector = data_driven_k(range(2, 6))
        similarity = _block_similarity(tuple(range(1, 10)), block_size=3)
        assert selector(similarity).n_clusters == 3

    def test_selects_k1_for_a_corpus_with_no_real_structure(self):
        # Silhouette alone is undefined at k=1, which is why the gap-statistic gate exists.
        selector = data_driven_k(range(2, 6))
        similarity = _uniform_similarity(tuple(range(1, 13)), 0.4)
        assert selector(similarity).n_clusters == 1

    def test_a_method_built_from_it_reports_one_cluster_for_no_structure(self):
        method = SpectralClusteringMethod(
            name="no-structure", description="test", k_selector=data_driven_k(range(2, 6))
        )
        similarity = _uniform_similarity(tuple(range(1, 13)), 0.4)
        result = method.compute(similarity)
        assert result.n_clusters == 1
        assert len(set(result.labels)) == 1

    def test_no_diagnostics_reported_for_a_k1_no_structure_choice(self):
        # partition_p_value/stability describe the silhouette-based choice among k>=2 - meaningless.
        selector = data_driven_k(range(2, 6))
        similarity = _uniform_similarity(tuple(range(1, 13)), 0.4)
        choice = selector(similarity)
        assert choice.partition_p_value is None
        assert choice.stability is None

    def test_a_method_built_from_it_reports_the_data_chosen_k_not_a_fixed_one(self):
        # The whole point of a data-driven selector: the caller never states 3 anywhere - it's.
        method = SpectralClusteringMethod(
            name="data-driven", description="test", k_selector=data_driven_k(range(2, 6))
        )
        similarity = _block_similarity(tuple(range(1, 10)), block_size=3)
        result = method.compute(similarity)
        assert result.n_clusters == 3
        assert len(set(result.labels)) == 3

    def test_reports_a_low_partition_p_value_for_well_separated_blocks(self):
        selector = data_driven_k(range(2, 6))
        similarity = _block_similarity(tuple(range(1, 10)), block_size=3)
        choice = selector(similarity)
        assert choice.partition_p_value is not None
        assert choice.partition_p_value < 0.05

    def test_reports_high_stability_for_well_separated_blocks(self):
        selector = data_driven_k(range(2, 6))
        similarity = _block_similarity(tuple(range(1, 10)), block_size=3)
        choice = selector(similarity)
        assert choice.stability is not None
        assert choice.stability > 0.5

    def test_a_method_built_from_it_threads_diagnostics_onto_the_result(self):
        method = SpectralClusteringMethod(
            name="data-driven", description="test", k_selector=data_driven_k(range(2, 6))
        )
        similarity = _block_similarity(tuple(range(1, 10)), block_size=3)
        result = method.compute(similarity)
        assert result.partition_p_value is not None
        assert result.k_stability is not None
