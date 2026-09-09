"""2D spectral embedding of psalm similarity for the Cluster page's scatter plot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tehillim_cluster.similarity import SimilarityResult


@dataclass(frozen=True, slots=True)
class Embedding2D:
    """A two-dimensional projection of the psalms for display, one point per psalm."""

    method: str
    psalm_numbers: tuple[int, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    structure_captured: float
    """Fraction of the corpus's total cluster-relevant spectral structure
    these 2 dimensions capture: sum of `(1 - eigenvalue)^2` for the 2
    eigenvalues used, over that same sum across every non-trivial
    eigenvalue. Analogous to classical MDS's "variance explained," for
    the Laplacian's opposite convention (a small eigenvalue is the
    meaningful one). 1.0 for a corpus with no cluster-relevant structure
    at all."""


def compute_embedding(similarity: SimilarityResult) -> Embedding2D:
    """Spectral embedding of `similarity.matrix`'s normalized graph Laplacian."""
    affinity = similarity.matrix
    n = affinity.shape[0]

    degree = affinity.sum(axis=1)
    inv_sqrt_degree = np.diag(1.0 / np.sqrt(degree))
    normalized_affinity = inv_sqrt_degree @ affinity @ inv_sqrt_degree
    laplacian = np.eye(n) - normalized_affinity

    # eigh (not eig) because laplacian is symmetric by construction.
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    non_trivial = eigenvalues[1:]
    weights = (1.0 - non_trivial) ** 2
    total_weight = float(np.sum(weights))
    captured_weight = float(np.sum(weights[:2]))
    structure_captured = captured_weight / total_weight if total_weight > 0 else 1.0

    scale = 1.0 - eigenvalues[1:3]
    coords = eigenvectors[:, 1:3] * scale
    #: eigh pins an eigenvector only up to sign, so without this a rerun mirrors the plot.
    largest = np.argmax(np.abs(coords), axis=0)
    signs = np.sign(coords[largest, np.arange(coords.shape[1])])
    coords = coords * np.where(signs == 0.0, 1.0, signs)

    return Embedding2D(
        method=similarity.method,
        psalm_numbers=similarity.psalm_numbers,
        x=tuple(float(v) for v in coords[:, 0]),
        y=tuple(float(v) for v in coords[:, 1]),
        structure_captured=structure_captured,
    )
