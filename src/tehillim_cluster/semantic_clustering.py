"""Derives one spectral clustering method per semantic similarity signal."""

from __future__ import annotations

from typing import Protocol

from tehillim_cluster.clustering import SpectralClusteringMethod, data_driven_k
from tehillim_cluster.k_selection import DEFAULT_K_VALUES


class DescribedMethod(Protocol):
    """The name and prose a clustering method inherits from the similarity it partitions."""

    @property
    def name(self) -> str:
        """The name this method is reported under."""

    @property
    def description(self) -> str:
        """What the method computes, for whoever reads the output."""


#: Search range for the data-driven k-selector, matching cluster_methods.py's _K_VALUES.
_K_VALUES = DEFAULT_K_VALUES

#: A similarity is named for its measure, its clustering for the partition it produces.
_COSINE_SUFFIX = "-cosine"
_SPECTRAL_SUFFIX = "-spectral"


def clustering_name_for(similarity_name: str) -> str:
    """The clustering name that pairs with a similarity name, refusing one it cannot pair with."""
    if not similarity_name.endswith(_COSINE_SUFFIX):
        raise ValueError(
            f"{similarity_name} does not end in {_COSINE_SUFFIX}: its clustering name is undefined"
        )
    return similarity_name.removesuffix(_COSINE_SUFFIX) + _SPECTRAL_SUFFIX


def clustering_for(similarity: DescribedMethod) -> SpectralClusteringMethod:
    """The spectral clustering that partitions one semantic similarity's matrix."""
    return SpectralClusteringMethod(
        name=clustering_name_for(similarity.name),
        description=similarity.description,
        k_selector=data_driven_k(_K_VALUES),
    )
