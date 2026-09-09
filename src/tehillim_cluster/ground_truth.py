"""Known structural and scholarly landmarks of the Psalter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TwinPsalmPair:
    """Psalms whose text substantially overlaps or duplicates."""

    psalms: tuple[int, ...]
    description: str


TWIN_PSALMS: tuple[TwinPsalmPair, ...] = (
    TwinPsalmPair(
        (14, 53),
        "Near-identical compositions, differing mainly in divine-name usage (YHWH vs. Elohim).",
    ),
    TwinPsalmPair((40, 70), "Psalm 40:14-18 recurs as the whole of Psalm 70."),
    TwinPsalmPair((57, 108), "Psalm 57:8-12 recurs as Psalm 108:2-6."),
    TwinPsalmPair((60, 108), "Psalm 60:7-14 recurs as Psalm 108:7-14."),
    # Psalm 18 is also a well-known twin of 2 Samuel 22.
)

#: Psalms 42-83: characteristically prefer "Elohim" over "YHWH" as the divine name.
ELOHISTIC_PSALTER = range(42, 84)

#: Psalms 120-134: each superscribed "שיר המעלות" (Song of Ascents) in the Masoretic Text.
SONGS_OF_ASCENT = range(120, 135)

#: Psalms 113-118: the traditional Passover/festival "Egyptian Hallel".
HALLEL_EGYPTIAN = range(113, 119)

#: Psalms 146-150: the Psalter's closing "Final Hallel".
HALLEL_FINAL = range(146, 151)


@dataclass(frozen=True, slots=True)
class RefrainPsalm:
    """A psalm (or split pair) built around one or more repeated refrain lines."""

    psalms: tuple[int, ...]
    refrain: str


REFRAIN_PSALMS: tuple[RefrainPsalm, ...] = (
    RefrainPsalm(
        (42, 43),
        "Why are you cast down, O my soul, and why are you disquieted "
        "within me? Hope in God (42:5, 11; 43:5).",
    ),
    RefrainPsalm(
        (46,),
        "The LORD of hosts is with us; the God of Jacob is our refuge (46:7, 11).",
    ),
    RefrainPsalm((56,), "In God I trust; I am not afraid (56:4, 11)."),
    RefrainPsalm(
        (80,),
        "Restore us, O God; let your face shine, that we may be saved (80:3, 7, 19).",
    ),
    RefrainPsalm((99,), "Holy is he (99:3, 5, 9)."),
    RefrainPsalm((136,), "For his steadfast love endures forever (every verse)."),
)

#: Psalms structured as (partial or complete) alphabetic acrostics.
ACROSTIC_PSALMS: tuple[int, ...] = (9, 10, 25, 34, 37, 111, 112, 119, 145)

#: Small, low-controversy exemplar sets for Gunkel's major genres, excluding cited blends.
GUNKEL_GENRE_EXEMPLARS: dict[str, tuple[int, ...]] = {
    "hymn": (8, 29, 33, 100, 103, 145, 146, 147, 148, 149, 150),
    "individual_lament": (3, 22, 38, 51, 88),
    "communal_lament": (44, 74, 79, 80, 137),
    "thanksgiving": (30, 34, 116, 118),
    "royal": (2, 18, 20, 21, 45, 72, 101, 110, 132, 144),
    "wisdom": (1, 37, 49, 127, 128),
}


@dataclass(frozen=True, slots=True)
class PsalterBook:
    """One of the Psalter's traditional five editorial "books"."""

    number: int
    range: range
    closing_doxology_psalm: int
    """Wilson's observation: each book closes with a doxology, marking an
    editorial seam, a narrower, better-evidenced claim than his fuller
    redactional thesis about royal-psalm placement at those seams."""


PSALTER_BOOKS: tuple[PsalterBook, ...] = (
    PsalterBook(1, range(1, 42), 41),
    PsalterBook(2, range(42, 73), 72),
    PsalterBook(3, range(73, 90), 89),
    PsalterBook(4, range(90, 107), 106),
    PsalterBook(5, range(107, 151), 150),
)
