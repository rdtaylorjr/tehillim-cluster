"""Joins tehillim-compare through published data, never through an import."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tehillim_cluster.compare_dataset import (
    ANALYSIS_COMPARE,
    PsalmFacts,
    psalm_core,
    read_psalms,
    read_similarities,
)

METHOD = "lexical-tfidf-cosine"


def _write_compare(root: Path, n: int = 6, method: str = METHOD) -> np.ndarray:
    raw = root / f"analysis={ANALYSIS_COMPARE}" / "domain=lexical" / "stage=raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows_m = rng.normal(size=(n, 4))
    rows_m /= np.linalg.norm(rows_m, axis=1, keepdims=True)
    matrix = (rows_m @ rows_m.T + 1) / 2
    np.fill_diagonal(matrix, 1.0)
    rows, cols = np.triu_indices(n, k=1)
    pq.write_table(
        pa.table(
            {
                "method": pa.array([method] * len(rows), type=pa.large_string()),
                "psalm_a": pa.array((rows + 1).astype("int32")),
                "psalm_b": pa.array((cols + 1).astype("int32")),
                "similarity": pa.array(matrix[rows, cols], type=pa.float64()),
            }
        ),
        raw / "similarity.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "method": pa.array([method], type=pa.large_string()),
                "description": pa.array(["Lexical TF-IDF cosine."], type=pa.large_string()),
            }
        ),
        raw / "methods.parquet",
    )
    proot = root / f"analysis={ANALYSIS_COMPARE}" / "stage=raw"
    proot.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "number": pa.array(np.arange(1, n + 1, dtype="int32")),
                "verse_count": pa.array(np.full(n, 6, dtype="int32")),
                "word_count": pa.array(np.full(n, 60, dtype="int32")),
                "incipit": pa.array(
                    [f"psalm {i}" for i in range(1, n + 1)], type=pa.large_string()
                ),
            }
        ),
        proot / "psalms.parquet",
    )
    return matrix


class TestReadPsalms:
    def test_it_reads_every_psalm_the_compare_run_covered(self, tmp_path: Path) -> None:
        _write_compare(tmp_path)

        assert [p.number for p in read_psalms(tmp_path)] == [1, 2, 3, 4, 5, 6]

    def test_it_carries_the_facts_a_payload_needs(self, tmp_path: Path) -> None:
        _write_compare(tmp_path)

        first = read_psalms(tmp_path)[0]

        assert (first.verse_count, first.word_count, first.incipit) == (6, 60, "psalm 1")

    def test_a_missing_compare_run_is_refused_with_the_remedy(self, tmp_path: Path) -> None:
        """Silently clustering nothing would publish an empty analysis that looks complete."""
        with pytest.raises(FileNotFoundError, match="tehillim-compare"):
            read_psalms(tmp_path)


class TestReadSimilarities:
    def test_it_rebuilds_a_symmetric_matrix_from_stored_pairs(self, tmp_path: Path) -> None:
        """Only the upper triangle is stored, so the lower half has to be mirrored back."""
        matrix = _write_compare(tmp_path)

        result = read_similarities(tmp_path)[0]

        assert np.allclose(result.matrix, matrix)

    def test_the_diagonal_comes_back_as_self_similarity(self, tmp_path: Path) -> None:
        _write_compare(tmp_path)

        assert np.allclose(np.diag(read_similarities(tmp_path)[0].matrix), 1.0)

    def test_it_carries_the_description_compare_published(self, tmp_path: Path) -> None:
        """The clustering inherits this prose, so losing it would leave the payload blank."""
        _write_compare(tmp_path)

        assert read_similarities(tmp_path)[0].description == "Lexical TF-IDF cosine."

    def test_psalm_numbers_survive_the_round_trip(self, tmp_path: Path) -> None:
        _write_compare(tmp_path)

        assert read_similarities(tmp_path)[0].psalm_numbers == (1, 2, 3, 4, 5, 6)

    def test_each_method_becomes_its_own_result(self, tmp_path: Path) -> None:
        _write_compare(tmp_path)
        raw = tmp_path / f"analysis={ANALYSIS_COMPARE}" / "domain=semantic" / "stage=raw"
        raw.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "method": pa.array(["a-cosine", "b-cosine"], type=pa.large_string()),
                    "psalm_a": pa.array([1, 1], type=pa.int32()),
                    "psalm_b": pa.array([2, 2], type=pa.int32()),
                    "similarity": pa.array([0.5, 0.7], type=pa.float64()),
                }
            ),
            raw / "similarity.parquet",
        )

        assert {r.method for r in read_similarities(tmp_path)} == {METHOD, "a-cosine", "b-cosine"}

    def test_a_missing_compare_run_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="tehillim-compare"):
            read_similarities(tmp_path)


class TestPsalmCore:
    def test_it_exports_the_camel_case_keys_the_payload_uses(self) -> None:
        facts = PsalmFacts(number=3, verse_count=8, word_count=71, incipit="why do the nations")

        assert psalm_core(facts) == {
            "number": 3,
            "verseCount": 8,
            "wordCount": 71,
            "incipit": "why do the nations",
        }
