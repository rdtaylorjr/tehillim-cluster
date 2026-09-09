"""Integration tests against the real BHSA corpus."""

from __future__ import annotations

import pytest

from tehillim_cluster import ground_truth as gt

pytestmark = pytest.mark.integration


def _cluster_of(clustering, psalm: int) -> int:
    index = clustering.psalm_numbers.index(psalm)
    return clustering.labels[index]


# --- Structural sanity, every signal ---------------------------------------


_ALL_CLUSTERING_FIXTURES = [
    "lexical_clustering",
    "root_clustering",
    "named_entity_identity_clustering",
    "verb_morphology_clustering",
    "person_profile_clustering",
    "lexical_set_clustering",
    "named_entity_clustering",
    "clause_type_clustering",
    "text_type_clustering",
    "clause_relation_clustering",
    "verb_sense_clustering",
]


@pytest.mark.parametrize("fixture_name", _ALL_CLUSTERING_FIXTURES)
def test_every_clustering_covers_all_150_psalms(fixture_name, request):
    clustering = request.getfixturevalue(fixture_name)
    assert clustering.psalm_numbers == tuple(range(1, 151))
    assert len(clustering.labels) == 150


@pytest.mark.parametrize("fixture_name", _ALL_CLUSTERING_FIXTURES)
def test_every_clustering_uses_a_data_chosen_cluster_count(fixture_name, request):
    # Each signal picks its own k via data_driven_k(cluster_methods._K_VALUES).
    clustering = request.getfixturevalue(fixture_name)
    assert 1 <= clustering.n_clusters <= 10
    assert all(0 <= label < clustering.n_clusters for label in clustering.labels)


# --- Twin psalms: the cleanest possible correctness check -------------------


def test_near_total_duplicate_twins_share_a_cluster_under_verb_morphology_and_clause_type(
    verb_morphology_clustering, clause_type_clustering
):
    # Checked across all four TWIN_PSALMS pairs: only the two near-duplicate pairs qualify.
    near_total_duplicate_pairs = [pair for pair in gt.TWIN_PSALMS if pair.psalms == (14, 53)] + [
        pair for pair in gt.TWIN_PSALMS if pair.psalms == (57, 108)
    ]
    assert len(near_total_duplicate_pairs) == 2

    for pair in near_total_duplicate_pairs:
        for clustering in (verb_morphology_clustering, clause_type_clustering):
            labels = {_cluster_of(clustering, psalm) for psalm in pair.psalms}
            assert len(labels) == 1


# --- Individual vs. communal lament under person-profile clustering --------


def test_individual_laments_all_share_one_cluster(person_profile_clustering):
    individual = gt.GUNKEL_GENRE_EXEMPLARS["individual_lament"]
    labels = {_cluster_of(person_profile_clustering, psalm) for psalm in individual}
    assert len(labels) == 1


def test_communal_laments_do_not_share_the_individual_lament_cluster(person_profile_clustering):
    individual = gt.GUNKEL_GENRE_EXEMPLARS["individual_lament"]
    communal = gt.GUNKEL_GENRE_EXEMPLARS["communal_lament"]
    individual_cluster = _cluster_of(person_profile_clustering, individual[0])
    communal_clusters = {_cluster_of(person_profile_clustering, psalm) for psalm in communal}
    assert individual_cluster not in communal_clusters


# --- Final Hallel under verb-morphology clustering: the honest result -----


def test_final_hallel_splits_into_two_coherent_subgroups(verb_morphology_clustering):
    # Checked empirically: 146/147/149 land together, 148/150 land together - two real subgroups.
    labels = {psalm: _cluster_of(verb_morphology_clustering, psalm) for psalm in gt.HALLEL_FINAL}
    assert labels[146] == labels[147] == labels[149]
    assert labels[148] == labels[150]
    assert labels[146] != labels[148]


def test_final_hallel_purest_imperative_pair_shares_a_cluster(verb_morphology_clustering):
    # Psalm 148 and 150 are the two shortest.
    assert _cluster_of(verb_morphology_clustering, 148) == _cluster_of(
        verb_morphology_clustering, 150
    )
