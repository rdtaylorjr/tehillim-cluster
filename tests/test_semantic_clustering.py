"""One clustering per semantic similarity, derived rather than hand-written eighty times."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tehillim_cluster.semantic_clustering import clustering_for, clustering_name_for


@dataclass(frozen=True)
class _Signal:
    name: str
    description: str


class TestClusteringNameFor:
    def test_a_cosine_measure_becomes_a_spectral_partition(self) -> None:
        assert clustering_name_for("miqrabert-mean-pool-cosine") == "miqrabert-mean-pool-spectral"

    def test_only_the_trailing_measure_is_replaced(self) -> None:
        """A model whose own name contains the word must keep it."""
        assert clustering_name_for("cosine-model-soft-alignment-cosine") == (
            "cosine-model-soft-alignment-spectral"
        )

    def test_a_name_it_cannot_pair_with_is_refused(self) -> None:
        """Inventing a name would ship a method nothing else references."""
        with pytest.raises(ValueError, match="euclidean"):
            clustering_name_for("some-model-euclidean")


class TestClusteringFor:
    def test_it_inherits_the_signals_description(self) -> None:
        """The pair describes one signal, so duplicating the prose would let the two drift."""
        signal = _Signal("a-cosine", "prose about a")

        assert clustering_for(signal).description == "prose about a"

    def test_it_names_the_partition_after_the_measure(self) -> None:
        assert clustering_for(_Signal("a-cosine", "")).name == "a-spectral"

    def test_each_signal_gets_its_own_clustering(self) -> None:
        first = clustering_for(_Signal("a-cosine", ""))
        second = clustering_for(_Signal("b-cosine", ""))

        assert first.name != second.name

    def test_it_carries_a_k_selector(self) -> None:
        assert clustering_for(_Signal("a-cosine", "")).k_selector is not None
