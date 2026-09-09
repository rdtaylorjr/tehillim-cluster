from __future__ import annotations

import pytest

from tehillim_cluster.clustering import ClusteringResult
from tehillim_cluster.genre_alignment import (
    _match_clusters_to_categories,
    compute_family_alignment,
    compute_genre_alignment,
)
from tehillim_cluster.gunkel_genre_index import GUNKEL_GENRES


def _clustering(psalm_numbers: tuple[int, ...], labels: tuple[int, ...], n_clusters: int):
    return ClusteringResult(
        method="test-spectral",
        description="test",
        psalm_numbers=psalm_numbers,
        labels=labels,
        n_clusters=n_clusters,
    )


def test_genres_match_the_canonical_list_and_order():
    clustering = _clustering((1,), (0,), 2)
    alignment = compute_genre_alignment(clustering)
    assert alignment.genres == GUNKEL_GENRES


def test_counts_a_psalm_into_its_genre_and_assigned_cluster():
    # Psalm 1 is Wisdom Psalm, Psalm 8 is Hymn (gunkel_genre_index.py).
    clustering = _clustering((1, 8), (0, 1), 2)
    alignment = compute_genre_alignment(clustering)

    wisdom_row = alignment.genres.index("Wisdom Psalm")
    hymn_row = alignment.genres.index("Hymn")
    assert alignment.counts[wisdom_row] == (1, 0)
    assert alignment.counts[hymn_row] == (0, 1)


def test_genre_totals_sum_each_row():
    # Psalms 3 and 6 are both Individual Lament.
    clustering = _clustering((3, 6), (0, 1), 2)
    alignment = compute_genre_alignment(clustering)
    lament_row = alignment.genres.index("Individual Lament")
    assert alignment.genre_totals[lament_row] == 2
    assert sum(alignment.counts[lament_row]) == alignment.genre_totals[lament_row]


def test_skips_psalms_excluded_from_the_gunkel_index():
    # Psalm 27 is excluded (SPLIT) - must not appear in any count.
    clustering = _clustering((27,), (0,), 1)
    alignment = compute_genre_alignment(clustering)
    assert sum(alignment.genre_totals) == 0


def test_every_included_psalm_is_counted_exactly_once():
    psalm_numbers = tuple(range(1, 151))
    labels = tuple(p % 6 for p in psalm_numbers)
    clustering = _clustering(psalm_numbers, labels, 6)
    alignment = compute_genre_alignment(clustering)
    # 150 psalms minus the 6 excluded composite/partial ones.
    assert sum(alignment.genre_totals) == 144


# --- External validation scores (purity, NMI, ARI) -------------------------

# 4 Individual Laments (3, 5, 6, 7) and 4 Hymns (8, 19, 29, 33) - two clean.
_LAMENT_AND_HYMN_PSALMS = (3, 5, 6, 7, 8, 19, 29, 33)


def test_perfect_clustering_scores_maximally_on_every_metric():
    # Laments in cluster 0, hymns in cluster 1 - clustering reproduces genre exactly.
    clustering = _clustering(_LAMENT_AND_HYMN_PSALMS, (0, 0, 0, 0, 1, 1, 1, 1), 2)
    alignment = compute_genre_alignment(clustering)
    assert alignment.purity == pytest.approx(1.0)
    assert alignment.ami == pytest.approx(1.0)
    assert alignment.ari == pytest.approx(1.0)


def test_a_clustering_independent_of_genre_scores_far_below_perfect():
    # Alternating assignment: each cluster is an even 50/50 lament/hymn mix.
    clustering = _clustering(_LAMENT_AND_HYMN_PSALMS, (0, 1, 0, 1, 0, 1, 0, 1), 2)
    alignment = compute_genre_alignment(clustering)
    assert alignment.purity == pytest.approx(0.5)
    assert alignment.ami < 0  # chance-corrected: below 0, not merely "not great"
    assert alignment.ari < 0  # worse than chance, not just "not great"


# --- AMI-permutation significance -------------------------------------------


def test_ami_p_value_is_low_for_a_perfect_clustering():
    clustering = _clustering(_LAMENT_AND_HYMN_PSALMS, (0, 0, 0, 0, 1, 1, 1, 1), 2)
    alignment = compute_genre_alignment(clustering)
    assert alignment.ami_p_value < 0.05


def test_ami_p_value_is_not_significant_for_a_genre_independent_clustering():
    clustering = _clustering(_LAMENT_AND_HYMN_PSALMS, (0, 1, 0, 1, 0, 1, 0, 1), 2)
    alignment = compute_genre_alignment(clustering)
    assert alignment.ami_p_value == pytest.approx(1.0)


def test_ami_p_value_is_within_valid_bounds():
    clustering = _clustering(_LAMENT_AND_HYMN_PSALMS, (0, 0, 0, 0, 1, 1, 1, 1), 2)
    alignment = compute_genre_alignment(clustering)
    assert 0.0 < alignment.ami_p_value <= 1.0


def test_ami_p_value_is_reproducible():
    clustering = _clustering(_LAMENT_AND_HYMN_PSALMS, (0, 0, 0, 0, 1, 1, 1, 1), 2)
    first = compute_genre_alignment(clustering)
    second = compute_genre_alignment(clustering)
    assert first.ami_p_value == second.ami_p_value


# --- Cluster-to-genre labeling (Hungarian algorithm) ------------------------


def test_matches_each_cluster_to_the_genre_it_overlaps_with_most():
    clustering = _clustering(_LAMENT_AND_HYMN_PSALMS, (0, 0, 0, 0, 1, 1, 1, 1), 2)
    alignment = compute_genre_alignment(clustering)
    assert alignment.cluster_genre_labels == ("Individual Lament", "Hymn")


def test_avoids_assigning_two_clusters_to_the_same_genre():
    # Cluster 0 overlaps Wisdom=10, Hymn=0; cluster 1 overlaps Wisdom=8, Hymn=5.
    counts = [[0, 0] for _ in GUNKEL_GENRES]
    wisdom_row = GUNKEL_GENRES.index("Wisdom Psalm")
    hymn_row = GUNKEL_GENRES.index("Hymn")
    counts[wisdom_row] = [10, 8]
    counts[hymn_row] = [0, 5]

    labels = _match_clusters_to_categories(counts, GUNKEL_GENRES, n_clusters=2)
    assert labels == ("Wisdom Psalm", "Hymn")


def test_leaves_a_cluster_unlabeled_if_it_shares_no_psalms_with_any_genre():
    counts = [[0, 0] for _ in GUNKEL_GENRES]
    counts[GUNKEL_GENRES.index("Wisdom Psalm")] = [5, 0]
    labels = _match_clusters_to_categories(counts, GUNKEL_GENRES, n_clusters=2)
    assert labels == ("Wisdom Psalm", None)


# --- Family (coarse, 6-category) alignment ----------------------------------


def test_family_alignment_uses_the_six_family_categories():
    from tehillim_cluster.gunkel_genre_index import GUNKEL_FAMILIES

    clustering = _clustering((1,), (0,), 2)
    alignment = compute_family_alignment(clustering)
    assert alignment.genres == GUNKEL_FAMILIES


def test_family_alignment_groups_genres_sharing_a_family():
    # Psalm 8 = Hymn, Psalm 47 = Enthronement Psalm - both fold into the Hymn family.
    clustering = _clustering((8, 47), (0, 0), 1)
    alignment = compute_family_alignment(clustering)
    hymn_row = alignment.genres.index("Hymn")
    assert alignment.genre_totals[hymn_row] == 2
