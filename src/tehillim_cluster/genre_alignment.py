"""Cross-tabulates a clustering against Gunkel's classification at two granularities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from tehillim_cluster.clustering import ClusteringResult
from tehillim_cluster.gunkel_genre_index import (
    GENRE_FAMILY,
    GUNKEL_FAMILIES,
    GUNKEL_GENRE_INDEX,
    GUNKEL_GENRES,
    GunkelClassification,
)
from tehillim_cluster.k_selection import DEFAULT_N_PERMUTATIONS


@dataclass(frozen=True, slots=True)
class GenreAlignment:
    """Contingency table cross-tabulating `clustering` against a Gunkel categorization."""

    genres: tuple[str, ...]
    """Rows, in canonical order (`GUNKEL_GENRES` or `GUNKEL_FAMILIES`)."""
    counts: tuple[tuple[int, ...], ...]
    """counts[category_index][cluster_index]"""
    genre_totals: tuple[int, ...]
    """counts[category_index] summed: how many indexed psalms carry that category."""
    cluster_genre_labels: tuple[str | None, ...]
    """Best-matching category per cluster index (Hungarian-algorithm
    one-to-one assignment, see module docstring), or None where no
    category with nonzero overlap remained available."""
    purity: float
    """Sum, over clusters, of that cluster's largest single-category
    count, divided by the number of indexed psalms. Unlike
    `cluster_genre_labels`, clusters may share a majority category here."""
    ami: float
    """Adjusted Mutual Information between cluster assignment and Gunkel
    category, over indexed psalms. ~0 for random clustering, 1 for exact
    agreement, can go slightly negative like ARI."""
    ari: float
    """Adjusted Rand Index, same scope: agreement on which psalm pairs
    are grouped together, corrected for chance."""
    ami_p_value: float
    """Permutation-test p-value for `ami`: the fraction of random
    relabelings (same cluster sizes, Gunkel categories fixed) whose AMI
    meets or exceeds the observed one. Not yet corrected across every
    shipped signal, see README.md's "Statistical validation methodology"
    for the cross-signal Benjamini-Hochberg correction."""


def compute_genre_alignment(
    clustering: ClusteringResult, *, ami_permutations: int = DEFAULT_N_PERMUTATIONS
) -> GenreAlignment:
    """Cross-tabulates `clustering` against Gunkel's 14-genre decomposition."""
    return _cross_tabulate(
        clustering,
        categories=GUNKEL_GENRES,
        category_of_entry=lambda entry: entry.genre,
        ami_permutations=ami_permutations,
    )


def compute_family_alignment(
    clustering: ClusteringResult, *, ami_permutations: int = DEFAULT_N_PERMUTATIONS
) -> GenreAlignment:
    """Cross-tabulates `clustering` against Gunkel's six top-level chapter families."""
    return _cross_tabulate(
        clustering,
        categories=GUNKEL_FAMILIES,
        category_of_entry=lambda entry: GENRE_FAMILY[entry.genre],
        ami_permutations=ami_permutations,
    )


def _cross_tabulate(
    clustering: ClusteringResult,
    *,
    categories: tuple[str, ...],
    category_of_entry: Callable[[GunkelClassification], str],
    ami_permutations: int = DEFAULT_N_PERMUTATIONS,
) -> GenreAlignment:
    """Skips psalms with no primary Gunkel classification entirely."""
    label_by_psalm = dict(zip(clustering.psalm_numbers, clustering.labels, strict=True))
    category_index = {category: i for i, category in enumerate(categories)}

    counts = [[0] * clustering.n_clusters for _ in categories]
    true_categories: list[str] = []
    predicted_clusters: list[int] = []
    for entry in GUNKEL_GENRE_INDEX:
        label = label_by_psalm.get(entry.psalm)
        if label is None:
            continue
        category = category_of_entry(entry)
        counts[category_index[category]][label] += 1
        true_categories.append(category)
        predicted_clusters.append(label)

    category_totals = tuple(sum(row) for row in counts)
    total_psalms = len(true_categories)

    n_categories = len(categories)
    cluster_majorities = (
        max(counts[cat_i][cluster_i] for cat_i in range(n_categories))
        for cluster_i in range(clustering.n_clusters)
    )
    purity = sum(cluster_majorities) / total_psalms if total_psalms else 0.0

    cluster_labels = _match_clusters_to_categories(counts, categories, clustering.n_clusters)
    ami = adjusted_mutual_info_score(true_categories, predicted_clusters)

    return GenreAlignment(
        genres=categories,
        counts=tuple(tuple(row) for row in counts),
        genre_totals=category_totals,
        cluster_genre_labels=cluster_labels,
        purity=purity,
        ami=ami,
        ari=adjusted_rand_score(true_categories, predicted_clusters),
        ami_p_value=_permutation_test_ami(
            true_categories, predicted_clusters, observed=ami, n_permutations=ami_permutations
        ),
    )


def _permutation_test_ami(
    true_categories: list[str],
    predicted_clusters: list[int],
    *,
    observed: float,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
) -> float:
    """Tests whether `observed` AMI beats a random same-size-distribution relabeling by chance."""
    if len(set(predicted_clusters)) < 2:
        #: A single-cluster "partition" is what data_driven_k returns when it finds no structure.
        return 1.0

    rng = np.random.default_rng(seed)
    predicted = np.array(predicted_clusters)
    null_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(predicted)
        null_scores[i] = adjusted_mutual_info_score(true_categories, shuffled)

    exceedances = int(np.sum(null_scores >= observed))
    return (exceedances + 1) / (n_permutations + 1)


def _match_clusters_to_categories(
    counts: list[list[int]], categories: tuple[str, ...], n_clusters: int
) -> tuple[str | None, ...]:
    """Solves linear-sum-assignment for a one-to-one cluster-to-category labeling."""
    n_categories = len(categories)
    cost = np.array(
        [
            [-counts[cat_i][cluster_i] for cat_i in range(n_categories)]
            for cluster_i in range(n_clusters)
        ]
    )
    cluster_rows, category_cols = linear_sum_assignment(cost)

    labels: list[str | None] = [None] * n_clusters
    for cluster_index, category_col in zip(cluster_rows, category_cols, strict=True):
        if counts[category_col][cluster_index] > 0:
            labels[cluster_index] = categories[category_col]
    return tuple(labels)
