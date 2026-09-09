from __future__ import annotations

import numpy as np
import pytest

from tehillim_cluster.embedding import compute_embedding
from tehillim_cluster.similarity import SimilarityResult


def _block_similarity(psalm_numbers: tuple[int, ...], block_size: int) -> SimilarityResult:
    """Well-separated blocks that a correct 2D layout should place far apart."""
    n = len(psalm_numbers)
    matrix = np.full((n, n), 0.05)
    for start in range(0, n, block_size):
        matrix[start : start + block_size, start : start + block_size] = 0.9
    np.fill_diagonal(matrix, 1.0)
    return SimilarityResult(
        method="test-similarity", description="test", psalm_numbers=psalm_numbers, matrix=matrix
    )


def _identical_similarity(psalm_numbers: tuple[int, ...]) -> SimilarityResult:
    n = len(psalm_numbers)
    matrix = np.ones((n, n))
    return SimilarityResult(
        method="test-similarity", description="test", psalm_numbers=psalm_numbers, matrix=matrix
    )


def test_records_the_source_method_name_and_psalm_numbers():
    similarity = _block_similarity(tuple(range(1, 9)), block_size=4)
    embedding = compute_embedding(similarity)
    assert embedding.method == "test-similarity"
    assert embedding.psalm_numbers == tuple(range(1, 9))


def test_coordinate_lists_match_psalm_count():
    similarity = _block_similarity(tuple(range(1, 9)), block_size=4)
    embedding = compute_embedding(similarity)
    assert len(embedding.x) == 8
    assert len(embedding.y) == 8


def test_well_separated_blocks_land_far_apart_in_2d():
    psalm_numbers = tuple(range(1, 9))
    similarity = _block_similarity(psalm_numbers, block_size=4)
    embedding = compute_embedding(similarity)
    points = np.column_stack([embedding.x, embedding.y])
    first_block_centroid = points[:4].mean(axis=0)
    second_block_centroid = points[4:].mean(axis=0)
    between_block_distance = np.linalg.norm(first_block_centroid - second_block_centroid)
    within_block_spread = max(
        np.linalg.norm(points[:4] - first_block_centroid, axis=1).max(),
        np.linalg.norm(points[4:] - second_block_centroid, axis=1).max(),
    )
    assert between_block_distance > 5 * within_block_spread


def test_uniformly_similar_psalms_collapse_to_one_point():
    # No cluster-relevant structure at all: every non-trivial normalized- Laplacian eigenvalue sits.
    psalm_numbers = tuple(range(1, 6))
    similarity = _identical_similarity(psalm_numbers)
    embedding = compute_embedding(similarity)
    points = np.column_stack([embedding.x, embedding.y])
    assert np.allclose(points, 0.0, atol=1e-8)


def test_deterministic_across_repeated_calls():
    similarity = _block_similarity(tuple(range(1, 9)), block_size=4)
    first = compute_embedding(similarity)
    second = compute_embedding(similarity)
    assert first.x == second.x
    assert first.y == second.y


def test_uses_the_same_eigenspace_spectral_clustering_partitions():
    # The whole point of switching away from classical MDS: for a signal whose data-driven k is 2.
    psalm_numbers = tuple(range(1, 9))
    similarity = _block_similarity(psalm_numbers, block_size=4)
    embedding = compute_embedding(similarity)
    signs = [1 if x > 0 else -1 for x in embedding.x]
    assert len(set(signs[:4])) == 1
    assert len(set(signs[4:])) == 1
    assert signs[0] != signs[4]


class TestStructureCaptured:
    def test_two_clean_blocks_are_almost_entirely_captured_by_2d(self):
        # A single real separating axis (two tight blocks) should leave essentially nothing for a.
        similarity = _block_similarity(tuple(range(1, 9)), block_size=4)
        embedding = compute_embedding(similarity)
        assert embedding.structure_captured > 0.95

    def test_is_exactly_one_for_a_perfectly_uniform_corpus(self):
        # No cluster-relevant structure exists at all.
        similarity = _identical_similarity(tuple(range(1, 6)))
        embedding = compute_embedding(similarity)
        assert embedding.structure_captured == pytest.approx(1.0)

    def test_is_bounded_between_zero_and_one(self):
        similarity = _block_similarity(tuple(range(1, 13)), block_size=3)
        embedding = compute_embedding(similarity)
        assert 0.0 <= embedding.structure_captured <= 1.0

    def test_lower_when_more_independent_blocks_compete_for_two_axes(self):
        # 4 well-separated blocks need more than 2 real separating axes.
        two_blocks = compute_embedding(_block_similarity(tuple(range(1, 9)), block_size=4))
        four_blocks = compute_embedding(_block_similarity(tuple(range(1, 13)), block_size=3))
        assert four_blocks.structure_captured < two_blocks.structure_captured
