from __future__ import annotations

from tehillim_cluster.gunkel_genre_index import (
    GENRE_FAMILY,
    GUNKEL_EXCLUDED_PSALMS,
    GUNKEL_FAMILIES,
    GUNKEL_GENRE_INDEX,
    GUNKEL_GENRES,
    family_of,
    genre_of,
)


def test_every_indexed_psalm_number_is_unique():
    psalms = [entry.psalm for entry in GUNKEL_GENRE_INDEX]
    assert len(psalms) == len(set(psalms))


def test_indexed_and_excluded_psalms_together_cover_all_150_exactly_once():
    indexed = {entry.psalm for entry in GUNKEL_GENRE_INDEX}
    assert indexed.isdisjoint(GUNKEL_EXCLUDED_PSALMS)
    assert indexed | GUNKEL_EXCLUDED_PSALMS == set(range(1, 151))


def test_every_entry_uses_a_canonical_genre():
    for entry in GUNKEL_GENRE_INDEX:
        assert entry.genre in GUNKEL_GENRES, f"Psalm {entry.psalm}: unknown genre {entry.genre!r}"


def test_six_composite_or_partial_psalms_are_excluded():
    # See the module docstring: Gunkel's own scheme splits these verse-by-verse across genres or.
    assert frozenset({27, 40, 66, 89, 96, 144}) == GUNKEL_EXCLUDED_PSALMS


def test_genre_of_returns_the_indexed_genre():
    assert genre_of(1) == "Wisdom Psalm"
    assert genre_of(150) == "Hymn"


def test_genre_of_returns_none_for_an_excluded_psalm():
    assert genre_of(27) is None


def test_hedged_entries_only_where_gunkel_himself_hedges():
    # Psalm 139's Hymn placement is debated in later reception (per the module docstring) but not.
    entry_139 = next(e for e in GUNKEL_GENRE_INDEX if e.psalm == 139)
    assert entry_139.hedged is False

    entry_58 = next(e for e in GUNKEL_GENRE_INDEX if e.psalm == 58)
    assert entry_58.hedged is True


def test_cross_listed_entries_use_the_first_non_hedged_option():
    # Psalm 106: "Communal Complaint (uncertain) / Confession (national) / Legend" - the first.
    entry_106 = next(e for e in GUNKEL_GENRE_INDEX if e.psalm == 106)
    assert entry_106.genre == "Confession (National)"
    assert entry_106.cross_listed is True
    assert entry_106.hedged is True


# --- Family grouping (coarse, 6-category view) ------------------------------


def test_every_genre_maps_to_a_canonical_family():
    for genre in GUNKEL_GENRES:
        assert genre in GENRE_FAMILY, f"{genre!r} has no family mapping"
        assert GENRE_FAMILY[genre] in GUNKEL_FAMILIES


def test_hymn_subtypes_share_the_hymn_family():
    assert GENRE_FAMILY["Hymn"] == "Hymn"
    assert GENRE_FAMILY["Enthronement Psalm"] == "Hymn"
    assert GENRE_FAMILY["Song of Zion"] == "Hymn"


def test_lament_and_communal_complaint_share_the_lament_family():
    assert GENRE_FAMILY["Individual Lament"] == "Lament"
    assert GENRE_FAMILY["Communal Complaint"] == "Lament"


def test_national_confession_folds_into_lament_not_its_own_family():
    # Both psalms carrying this label (81, 106) cite E section II.B.3.
    assert GENRE_FAMILY["Confession (National)"] == "Lament"


def test_family_of_returns_the_mapped_family():
    assert family_of(1) == "Wisdom Psalm"  # Psalm 1 is Wisdom Psalm
    assert family_of(150) == "Hymn"  # Psalm 150 is Hymn


def test_family_of_returns_none_for_an_excluded_psalm():
    assert family_of(27) is None
