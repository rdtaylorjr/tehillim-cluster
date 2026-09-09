"""eigh fixes an eigenvector only up to sign, so the published coordinates need a convention."""

from __future__ import annotations

import numpy as np

from tehillim_cluster.embedding import compute_embedding
from tehillim_cluster.similarity import SimilarityResult

N = 60


def _affinity(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(N, 12))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    matrix = (raw @ raw.T + (raw @ raw.T).T) / 2
    matrix = (matrix + 1.0) / 2.0
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _embed(matrix: np.ndarray):
    return compute_embedding(
        SimilarityResult(
            method="m", description="d", psalm_numbers=tuple(range(1, N + 1)), matrix=matrix
        )
    )


def _perturbed(matrix: np.ndarray, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=scale, size=(N, N))
    noise = (noise + noise.T) / 2
    perturbed = matrix + noise
    np.fill_diagonal(perturbed, 1.0)
    return perturbed


class TestTheSignIsPinned:
    def test_a_float32_sized_perturbation_does_not_mirror_the_plot(self) -> None:
        """Unpinned, a 1e-7 change flipped both axes, so a rerun's every point changed sign."""
        base = _affinity()
        first = _embed(base)

        for seed in range(8):
            second = _embed(_perturbed(base, 2.4e-07, seed))
            assert np.dot(np.array(first.x), np.array(second.x)) > 0
            assert np.dot(np.array(first.y), np.array(second.y)) > 0

    def test_negating_an_eigenvector_cannot_change_the_output(self) -> None:
        """The convention has to hold whichever sign LAPACK happens to return."""
        base = _affinity(seed=3)
        first = _embed(base)
        second = _embed(base.copy())

        assert first.x == second.x
        assert first.y == second.y

    def test_the_largest_magnitude_coordinate_is_positive_on_each_axis(self) -> None:
        embedding = _embed(_affinity(seed=5))

        for axis in (np.array(embedding.x), np.array(embedding.y)):
            assert axis[int(np.argmax(np.abs(axis)))] > 0

    def test_the_magnitudes_are_untouched_by_the_convention(self) -> None:
        """Pinning the sign must not move any point off its axis position."""
        base = _affinity(seed=7)
        embedding = _embed(base)

        affinity = base
        degree = affinity.sum(axis=1)
        inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        laplacian = np.eye(N) - inv_sqrt @ affinity @ inv_sqrt
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        expected = eigenvectors[:, 1:3] * (1.0 - eigenvalues[1:3])

        assert np.allclose(np.abs(embedding.x), np.abs(expected[:, 0]))
        assert np.allclose(np.abs(embedding.y), np.abs(expected[:, 1]))
