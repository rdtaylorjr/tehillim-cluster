"""Serializes `gunkel_genre_index.py` into gunkel.json."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from tehillim_cluster.gunkel_genre_index import (
    GENRE_FAMILY,
    GUNKEL_FAMILIES,
    GUNKEL_GENRE_INDEX,
    GUNKEL_GENRES,
)


def build_gunkel_payload() -> dict[str, Any]:
    """Assemble the full JSON-serializable Gunkel reference payload."""
    genre_by_psalm = {entry.psalm: entry.genre for entry in GUNKEL_GENRE_INDEX}
    return {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "genres": list(GUNKEL_GENRES),
        "families": list(GUNKEL_FAMILIES),
        "psalms": [
            {
                "number": psalm,
                "genre": genre_by_psalm.get(psalm),
                "family": GENRE_FAMILY[genre_by_psalm[psalm]] if psalm in genre_by_psalm else None,
            }
            for psalm in range(1, 151)
        ],
    }
