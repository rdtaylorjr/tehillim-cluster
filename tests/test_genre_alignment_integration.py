"""Integration test confirming compute_genre_alignment wires cleanly against real BHSA."""

from __future__ import annotations

import pytest

from tehillim_cluster.genre_alignment import compute_genre_alignment

pytestmark = pytest.mark.integration

#: Independently counted from gunkel_genre_index.GUNKEL_GENRE_INDEX.
EXPECTED_GENRE_TOTALS = {
    "Hymn": 25,
    "Enthronement Psalm": 4,
    "Song of Zion": 6,
    "Individual Lament": 45,
    "Communal Complaint": 7,
    "Individual Thanksgiving": 8,
    "Community Thanksgiving": 2,
    "Royal Psalm": 9,
    "Wisdom Psalm": 9,
    "Liturgy": 12,
    "Legend / Ancient Story": 1,
    "Confession (National)": 2,
    "Mixed Type": 9,
    "Miscellaneous": 5,
}


def test_genre_totals_match_the_gunkel_index_regardless_of_clustering(
    verb_morphology_clustering,
):
    alignment = compute_genre_alignment(verb_morphology_clustering)
    totals = dict(zip(alignment.genres, alignment.genre_totals, strict=True))
    assert totals == EXPECTED_GENRE_TOTALS
    assert sum(alignment.genre_totals) == 144


def test_counts_are_consistent_across_different_clusterings_of_the_same_psalms(
    verb_morphology_clustering, person_profile_clustering
):
    # Genre totals are a property of the Gunkel index alone.
    a = compute_genre_alignment(verb_morphology_clustering)
    b = compute_genre_alignment(person_profile_clustering)
    assert a.genre_totals == b.genre_totals


def test_validation_scores_are_real_but_modest_not_near_perfect(
    verb_morphology_clustering, person_profile_clustering
):
    # Checked against real computed output first.
    verb_alignment = compute_genre_alignment(verb_morphology_clustering)
    person_alignment = compute_genre_alignment(person_profile_clustering)

    assert 0.0 < verb_alignment.ami < 0.15
    assert 0.1 < person_alignment.ami < 0.3
    assert person_alignment.ami > verb_alignment.ami

    for alignment in (verb_alignment, person_alignment):
        assert 0.0 < alignment.ari < 0.5
        assert 0.3 < alignment.purity < 0.6


def test_verb_morphology_clustering_leaves_most_genres_without_a_cluster(
    verb_morphology_clustering,
):
    # Verb-morphology's data-chosen cluster count is far smaller than Gunkel's 14 genres.
    alignment = compute_genre_alignment(verb_morphology_clustering)
    assigned_genres = {label for label in alignment.cluster_genre_labels if label is not None}
    assert len(assigned_genres) <= verb_morphology_clustering.n_clusters
    assert len(assigned_genres) < len(alignment.genres)
