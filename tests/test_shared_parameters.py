"""The sweep bounds and permutation count every method shares, held to one definition."""

from __future__ import annotations

import ast
import pathlib

import pytest

from tehillim_cluster.k_selection import (
    DEFAULT_K_VALUES,
    DEFAULT_N_PERMUTATIONS,
    GAP_K_VALUES,
)

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "tehillim_cluster"


def test_the_silhouette_sweep_starts_at_two_clusters() -> None:
    """Silhouette is undefined for one cluster, so the shared sweep cannot offer k=1."""
    assert min(DEFAULT_K_VALUES) == 2


def test_the_gap_sweep_alone_includes_the_no_cluster_null() -> None:
    """The gap statistic scores k=1 as its reference, which is why its sweep differs."""
    assert min(GAP_K_VALUES) == 1


def test_both_sweeps_stop_at_the_same_largest_k() -> None:
    """Two sweeps ending at different k would compare selections over different candidate sets."""
    assert max(DEFAULT_K_VALUES) == max(GAP_K_VALUES)


def test_the_gap_sweep_covers_every_k_the_shared_sweep_does() -> None:
    assert set(DEFAULT_K_VALUES) <= set(GAP_K_VALUES)


def test_the_permutation_count_puts_the_p_value_floor_below_the_conventional_alpha() -> None:
    """An add-one p-value floors at 1/(B+1), which must sit below 0.05 to be reportable."""
    assert 1 / (DEFAULT_N_PERMUTATIONS + 1) < 0.05


@pytest.mark.parametrize("path", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_no_module_restates_the_shared_permutation_count(path: pathlib.Path) -> None:
    """A second copy could drift, and the payload's p-values would differ in resolution."""
    literals = [
        node.lineno
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and node.value == DEFAULT_N_PERMUTATIONS
    ]

    if path.name == "k_selection.py":
        pytest.skip("k_selection defines the constant, so it holds the only literal")

    assert literals == []
