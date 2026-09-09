"""Structural validation of the reference/ground-truth data itself."""

from __future__ import annotations

from tehillim_cluster import ground_truth as gt

ALL_PSALM_NUMBERS = range(1, 151)


def _assert_valid_psalm_numbers(numbers: object) -> None:
    for number in numbers:  # type: ignore[attr-defined]
        assert number in ALL_PSALM_NUMBERS, f"{number} is not a valid psalm number (1-150)"


class TestTwinPsalms:
    def test_nonempty(self):
        assert len(gt.TWIN_PSALMS) > 0

    def test_every_pair_has_at_least_two_psalms(self):
        for pair in gt.TWIN_PSALMS:
            assert len(pair.psalms) >= 2

    def test_every_pair_references_valid_psalm_numbers(self):
        for pair in gt.TWIN_PSALMS:
            _assert_valid_psalm_numbers(pair.psalms)

    def test_every_pair_has_no_duplicate_psalms_within_itself(self):
        for pair in gt.TWIN_PSALMS:
            assert len(set(pair.psalms)) == len(pair.psalms)

    def test_every_pair_has_a_nonempty_description(self):
        for pair in gt.TWIN_PSALMS:
            assert pair.description.strip()

    def test_known_documented_twins_are_present(self):
        # The specific pairs this pipeline's integration tests already rely on.
        pair_sets = [set(pair.psalms) for pair in gt.TWIN_PSALMS]
        assert {14, 53} in pair_sets
        assert any({57, 108} <= s for s in pair_sets)
        assert any({60, 108} <= s for s in pair_sets)


class TestElohisticPsalter:
    def test_matches_documented_boundary(self):
        assert range(42, 84) == gt.ELOHISTIC_PSALTER

    def test_all_members_are_valid_psalm_numbers(self):
        _assert_valid_psalm_numbers(gt.ELOHISTIC_PSALTER)


class TestSongsOfAscent:
    def test_matches_documented_boundary(self):
        assert range(120, 135) == gt.SONGS_OF_ASCENT

    def test_has_fifteen_psalms(self):
        assert len(gt.SONGS_OF_ASCENT) == 15


class TestHallel:
    def test_egyptian_hallel_matches_documented_boundary(self):
        assert range(113, 119) == gt.HALLEL_EGYPTIAN

    def test_final_hallel_matches_documented_boundary(self):
        assert range(146, 151) == gt.HALLEL_FINAL

    def test_egyptian_and_final_hallel_do_not_overlap(self):
        assert set(gt.HALLEL_EGYPTIAN).isdisjoint(set(gt.HALLEL_FINAL))


class TestRefrainPsalms:
    def test_nonempty(self):
        assert len(gt.REFRAIN_PSALMS) > 0

    def test_every_entry_references_valid_psalm_numbers(self):
        for entry in gt.REFRAIN_PSALMS:
            _assert_valid_psalm_numbers(entry.psalms)

    def test_every_entry_has_a_nonempty_refrain_description(self):
        for entry in gt.REFRAIN_PSALMS:
            assert entry.refrain.strip()

    def test_forty_two_and_forty_three_are_grouped_together(self):
        # They share one refrain and are widely read as a single split composition - the one case.
        assert any(set(entry.psalms) == {42, 43} for entry in gt.REFRAIN_PSALMS)


class TestAcrosticPsalms:
    def test_nonempty(self):
        assert len(gt.ACROSTIC_PSALMS) > 0

    def test_all_members_are_valid_psalm_numbers(self):
        _assert_valid_psalm_numbers(gt.ACROSTIC_PSALMS)

    def test_no_duplicates(self):
        assert len(set(gt.ACROSTIC_PSALMS)) == len(gt.ACROSTIC_PSALMS)

    def test_is_sorted(self):
        assert list(gt.ACROSTIC_PSALMS) == sorted(gt.ACROSTIC_PSALMS)

    def test_includes_the_well_known_acrostics(self):
        # 119 (the great acrostic) and 145 are uncontroversial; a data-entry slip that dropped them.
        assert 119 in gt.ACROSTIC_PSALMS
        assert 145 in gt.ACROSTIC_PSALMS


class TestGunkelGenreExemplars:
    def test_nonempty(self):
        assert len(gt.GUNKEL_GENRE_EXEMPLARS) > 0

    def test_every_category_has_at_least_three_exemplars(self):
        # Fewer than three gives no meaningful "within-group" comparison for a pairwise-similarity.
        for genre, psalms in gt.GUNKEL_GENRE_EXEMPLARS.items():
            assert len(psalms) >= 3, f"{genre} has too few exemplars for group comparison"

    def test_every_category_references_valid_psalm_numbers(self):
        for psalms in gt.GUNKEL_GENRE_EXEMPLARS.values():
            _assert_valid_psalm_numbers(psalms)

    def test_every_category_has_no_duplicate_psalms_within_itself(self):
        for genre, psalms in gt.GUNKEL_GENRE_EXEMPLARS.items():
            assert len(set(psalms)) == len(psalms), f"{genre} has a duplicate"

    def test_hymn_category_includes_the_final_hallel(self):
        # The genre-fingerprint hypothesis is motivated by the Final Hallel as a textbook hymn.
        hymn = set(gt.GUNKEL_GENRE_EXEMPLARS["hymn"])
        assert set(gt.HALLEL_FINAL) <= hymn


class TestPsalterBooks:
    def test_five_books(self):
        assert len(gt.PSALTER_BOOKS) == 5

    def test_books_are_numbered_one_through_five_in_order(self):
        assert [b.number for b in gt.PSALTER_BOOKS] == [1, 2, 3, 4, 5]

    def test_books_cover_1_to_150_with_no_gaps_or_overlap(self):
        covered: list[int] = []
        for book in gt.PSALTER_BOOKS:
            covered.extend(book.range)
        assert covered == list(range(1, 151))

    def test_each_closing_doxology_psalm_is_the_last_psalm_in_its_book(self):
        for book in gt.PSALTER_BOOKS:
            assert book.closing_doxology_psalm == book.range[-1]
