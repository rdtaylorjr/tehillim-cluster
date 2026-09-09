"""Reads the compare analysis this package clusters, so the two repos join through data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq

from tehillim_cluster.similarity import SimilarityResult

if TYPE_CHECKING:
    from pathlib import Path

#: The analysis this package reads, written by tehillim-compare.
ANALYSIS_COMPARE = "compare"

#: The analysis this package writes.
ANALYSIS_CLUSTER = "cluster"

#: Reference data both analyses read, so it belongs to neither of them.
REFERENCE_ROOT = "reference"

CORPUS_SOURCE = "BHSA via Text-Fabric, read through tehillim-compare"


@dataclass(frozen=True, slots=True)
class PsalmFacts:
    """The method-independent psalm facts a payload needs, read rather than recomputed."""

    number: int
    verse_count: int
    word_count: int
    incipit: str


def psalm_core(psalm: PsalmFacts) -> dict[str, object]:
    """Method-independent psalm facts, exported once and shared."""
    return {
        "number": psalm.number,
        "verseCount": psalm.verse_count,
        "wordCount": psalm.word_count,
        "incipit": psalm.incipit,
    }


def compare_root(data_root: Path) -> Path:
    """Where tehillim-compare publishes the analysis this package consumes."""
    return data_root / f"analysis={ANALYSIS_COMPARE}"


def read_psalms(data_root: Path) -> list[PsalmFacts]:
    """Every psalm the compare run covered, in corpus order."""
    path = compare_root(data_root) / "stage=raw" / "psalms.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no psalm facts at {path}: run tehillim-compare against this data root first"
        )
    table = pq.read_table(path).sort_by("number")
    return [
        PsalmFacts(number=int(n), verse_count=int(v), word_count=int(w), incipit=str(i))
        for n, v, w, i in zip(
            table.column("number").to_pylist(),
            table.column("verse_count").to_pylist(),
            table.column("word_count").to_pylist(),
            table.column("incipit").to_pylist(),
            strict=True,
        )
    ]


def _descriptions(domain_dir: Path) -> dict[str, str]:
    """Each method's prose, so a clustering inherits it rather than restating it."""
    path = domain_dir / "stage=raw" / "methods.parquet"
    if not path.exists():
        return {}
    table = pq.read_table(path)
    return dict(
        zip(
            table.column("method").to_pylist(),
            table.column("description").to_pylist(),
            strict=True,
        )
    )


def read_similarities(data_root: Path) -> list[SimilarityResult]:
    """Every similarity matrix the compare run published, rebuilt from its stored pairs."""
    root = compare_root(data_root)
    domains = sorted(root.glob("domain=*"))
    if not domains:
        raise FileNotFoundError(
            f"no compare domains under {root}: run tehillim-compare against this data root first"
        )
    results: list[SimilarityResult] = []
    for domain_dir in domains:
        path = domain_dir / "stage=raw" / "similarity.parquet"
        if not path.exists():
            continue
        descriptions = _descriptions(domain_dir)
        results.extend(_results_from(pq.read_table(path), descriptions))
    return results


def _results_from(table: object, descriptions: dict[str, str]) -> list[SimilarityResult]:
    """One SimilarityResult per method in a stored pair table."""
    methods = np.asarray(table.column("method").to_pylist())  # type: ignore[attr-defined]
    psalm_a = np.asarray(table.column("psalm_a").to_pylist(), dtype=np.int64)  # type: ignore[attr-defined]
    psalm_b = np.asarray(table.column("psalm_b").to_pylist(), dtype=np.int64)  # type: ignore[attr-defined]
    values = np.asarray(table.column("similarity").to_pylist(), dtype=np.float64)  # type: ignore[attr-defined]

    results = []
    for method in sorted(set(methods.tolist())):
        rows = methods == method
        numbers = np.unique(np.concatenate([psalm_a[rows], psalm_b[rows]]))
        index = {int(n): i for i, n in enumerate(numbers)}
        matrix = np.eye(len(numbers), dtype=np.float64)
        a = np.fromiter(
            (index[int(x)] for x in psalm_a[rows]), dtype=np.intp, count=int(rows.sum())
        )
        b = np.fromiter(
            (index[int(x)] for x in psalm_b[rows]), dtype=np.intp, count=int(rows.sum())
        )
        #: Stored as the upper triangle only, since the matrix it came from is symmetric.
        matrix[a, b] = values[rows]
        matrix[b, a] = values[rows]
        results.append(
            SimilarityResult(
                method=str(method),
                description=descriptions.get(str(method), ""),
                psalm_numbers=tuple(int(n) for n in numbers),
                matrix=matrix,
            )
        )
    return results
