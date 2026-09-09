"""The cluster analysis: partitioned Parquet in the same grammar the compare analysis uses."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tehillim_cluster.clustering import ClusteringResult
from tehillim_cluster.dataset import (
    ANALYSIS_CLUSTER,
    assignments_table,
    cluster_metrics_table,
    domain_of,
    embedding_table,
    partition_path,
    ui_path,
    write_cluster_dataset,
)
from tehillim_cluster.embedding import Embedding2D

LEXICAL = "lexical-spectral"
SEMANTIC = "miqrabert-mean-pool-spectral"


def _clustering(method: str = LEXICAL) -> ClusteringResult:
    return ClusteringResult(
        method=method,
        description="d",
        psalm_numbers=(1, 2, 3),
        labels=(0, 1, 1),
        n_clusters=2,
        partition_p_value=0.01,
        k_stability=0.9,
    )


def _embedding(method: str = LEXICAL) -> Embedding2D:
    return Embedding2D(
        method=method,
        psalm_numbers=(1, 2, 3),
        x=(0.1, 0.2, 0.3),
        y=(-0.1, -0.2, -0.3),
        structure_captured=0.8,
    )


def _method_payload(method: str = LEXICAL) -> dict:
    return {
        "id": method,
        "nClusters": 2,
        "partitionPValue": 0.01,
        "kStability": 0.9,
        "embedding": {"structureCaptured": 0.8},
        "genreAlignment": {
            "purity": 0.5,
            "ami": 0.4,
            "ari": 0.3,
            "amiPValue": 0.02,
            "amiPValueAdjusted": 0.04,
        },
        "familyAlignment": {
            "purity": 0.6,
            "ami": 0.5,
            "ari": 0.35,
            "amiPValue": 0.03,
            "amiPValueAdjusted": 0.06,
        },
    }


class TestDomainOf:
    def test_a_feature_partition_takes_its_signals_domain(self) -> None:
        assert domain_of(LEXICAL) == "lexical"

    def test_a_similarity_name_resolves_too(self) -> None:
        """Embeddings carry the similarity's name while partitions carry the spectral one."""
        assert domain_of("lexical-tfidf-cosine") == "lexical"

    def test_a_semantic_partition_is_semantic(self) -> None:
        assert domain_of(SEMANTIC) == "semantic"

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("verb-morphology-tfidf-cosine", "morphological"),
            ("person-profile-tfidf-cosine", "morphological"),
            ("clause-type-tfidf-cosine", "syntactic"),
            ("text-type-tfidf-cosine", "syntactic"),
            ("clause-relation-tfidf-cosine", "syntactic"),
        ],
    )
    def test_a_formal_signal_lands_in_the_shared_domain_vocabulary(
        self, method: str, expected: str
    ) -> None:
        """These names are the partition the other repos read, so a drift misfiles the output."""
        assert domain_of(method) == expected

    def test_an_unrecognisable_name_is_refused(self) -> None:
        """Bucketing a new method silently would hide it in the wrong partition."""
        with pytest.raises(KeyError, match="not-a-method"):
            domain_of("not-a-method")


class TestTables:
    def test_assignments_carry_each_psalms_cluster(self) -> None:
        rows = assignments_table([_clustering()]).to_pylist()

        assert {r["psalm"]: r["cluster"] for r in rows} == {1: 0, 2: 1, 3: 1}

    def test_embeddings_carry_both_coordinates(self) -> None:
        rows = embedding_table([_embedding()]).to_pylist()

        assert rows[0]["x"] == pytest.approx(0.1)
        assert rows[0]["y"] == pytest.approx(-0.1)

    def test_metrics_use_the_long_schema(self) -> None:
        table = cluster_metrics_table([_method_payload()])

        assert table.column_names == [
            "method",
            "domain",
            "metric",
            "value",
            "q_value",
            "q_value_by",
        ]

    def test_the_adjusted_p_value_rides_beside_the_raw_one(self) -> None:
        rows = cluster_metrics_table([_method_payload()]).to_pylist()
        genre_p = next(r for r in rows if r["metric"] == "genre_ami_p_value")

        assert genre_p["value"] == pytest.approx(0.02)
        assert genre_p["q_value"] == pytest.approx(0.04)


class TestWriteClusterDataset:
    def _written(self, tmp_path: Path) -> list[Path]:
        return write_cluster_dataset(
            tmp_path,
            clusterings=[_clustering(LEXICAL), _clustering(SEMANTIC)],
            embeddings=[_embedding(LEXICAL), _embedding(SEMANTIC)],
            cluster_methods=[_method_payload(LEXICAL), _method_payload(SEMANTIC)],
        )

    def test_each_domain_gets_its_own_partition(self, tmp_path: Path) -> None:
        written = self._written(tmp_path)

        assert {p.parent.parent.name for p in written} == {"domain=lexical", "domain=semantic"}

    def test_it_writes_every_stage(self, tmp_path: Path) -> None:
        assert {p.parent.name for p in self._written(tmp_path)} == {
            "stage=profiles",
            "stage=detail",
            "stage=master",
        }

    def test_everything_lands_under_the_cluster_analysis(self, tmp_path: Path) -> None:
        assert {p.parent.parent.parent.name for p in self._written(tmp_path)} == {
            f"analysis={ANALYSIS_CLUSTER}"
        }

    def test_a_methods_rows_stay_in_its_own_domain(self, tmp_path: Path) -> None:
        """A method leaking across partitions would double-count it in any roll-up."""
        self._written(tmp_path)
        table = pq.read_table(
            tmp_path / f"analysis={ANALYSIS_CLUSTER}/domain=lexical/stage=master"
            "/method_metrics_long.parquet"
        )

        assert set(table.column("method").to_pylist()) == {LEXICAL}

    def test_every_written_file_reads_back(self, tmp_path: Path) -> None:
        for path in self._written(tmp_path):
            assert pq.read_table(path).num_rows > 0


class TestUiPath:
    def test_the_payload_sits_under_its_analysis_without_a_domain(self, tmp_path: Path) -> None:
        assert ui_path(tmp_path, ANALYSIS_CLUSTER, "clustering.json") == (
            tmp_path / "analysis=cluster" / "stage=ui" / "clustering.json"
        )

    def test_partition_path_matches_the_shared_grammar(self, tmp_path: Path) -> None:
        assert partition_path(tmp_path, ANALYSIS_CLUSTER, "lexical", "master", "m.parquet") == (
            tmp_path / "analysis=cluster" / "domain=lexical" / "stage=master" / "m.parquet"
        )
