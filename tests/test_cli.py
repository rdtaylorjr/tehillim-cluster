"""The entry point reads a published compare analysis and writes the cluster one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tehillim_cluster.cli import (
    DEFAULT_CLUSTERING_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    default_cluster_output,
    default_gunkel_output,
    main,
    method_for,
    parse_args,
)
from tehillim_cluster.similarity import SimilarityResult


def _write_compare(root: Path, n: int = 24, method: str = "lexical-tfidf-cosine") -> None:
    raw = root / "analysis=compare" / "domain=lexical" / "stage=raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    m = rng.normal(size=(n, 5))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    matrix = (m @ m.T + 1) / 2
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
                "description": pa.array(["prose"], type=pa.large_string()),
            }
        ),
        raw / "methods.parquet",
    )
    proot = root / "analysis=compare" / "stage=raw"
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


class TestParseArgs:
    def test_the_data_root_defaults_into_this_repo(self) -> None:
        assert parse_args([], env={}).data_root == DEFAULT_DATA_ROOT

    def test_the_data_root_reads_the_environment(self) -> None:
        assert parse_args([], env={"TEHILLIM_DATA_DIR": "/data"}).data_root == Path("/data")

    def test_the_flag_overrides_the_environment(self) -> None:
        args = parse_args(["--data-root", "/flag"], env={"TEHILLIM_DATA_DIR": "/env"})

        assert args.data_root == Path("/flag")

    def test_the_payload_paths_default_to_none_so_they_follow_the_root(self) -> None:
        args = parse_args([], env={})

        assert (args.cluster_output, args.gunkel_output) == (None, None)

    def test_the_cache_dir_reads_the_environment(self) -> None:
        args = parse_args([], env={"TEHILLIM_CLUSTERING_CACHE_DIR": "/cache"})

        assert args.cache_dir == Path("/cache")

    def test_the_cache_dir_defaults_into_this_repo(self) -> None:
        assert parse_args([], env={}).cache_dir == DEFAULT_CLUSTERING_CACHE_DIR


class TestDefaultOutputs:
    def test_the_cluster_payload_sits_under_its_analysis(self) -> None:
        assert default_cluster_output(Path("/d")) == Path(
            "/d/analysis=cluster/stage=ui/clustering.json"
        )

    def test_the_gunkel_reference_sits_outside_either_analysis(self) -> None:
        """Both the compare and cluster pages read it, so it belongs to neither."""
        assert default_gunkel_output(Path("/d")) == Path("/d/reference/stage=ui/gunkel.json")


class TestMethodFor:
    def test_a_feature_signal_uses_its_configured_clustering(self) -> None:
        signal = SimilarityResult(
            method="lexical-tfidf-cosine", description="d", psalm_numbers=(1,), matrix=np.eye(1)
        )

        assert method_for(signal).name == "lexical-spectral"

    def test_a_semantic_signal_gets_a_derived_clustering(self) -> None:
        """Nothing configures the eighty semantic methods by hand any more."""
        signal = SimilarityResult(
            method="some-model-mean-pool-cosine",
            description="d",
            psalm_numbers=(1,),
            matrix=np.eye(1),
        )

        assert method_for(signal).name == "some-model-mean-pool-spectral"

    def test_the_derived_clustering_inherits_the_description(self) -> None:
        signal = SimilarityResult(
            method="a-cosine", description="published prose", psalm_numbers=(1,), matrix=np.eye(1)
        )

        assert method_for(signal).description == "published prose"


class TestMain:
    def test_it_publishes_the_cluster_analysis_from_a_compare_run(self, tmp_path: Path) -> None:
        _write_compare(tmp_path)

        assert main(["--data-root", str(tmp_path), "--cache-dir", str(tmp_path / "cache")]) == 0
        assert (tmp_path / "analysis=cluster/stage=ui/clustering.json").exists()
        assert (
            tmp_path / "analysis=cluster/domain=lexical/stage=profiles/assignments.parquet"
        ).exists()

    def test_it_refuses_a_data_root_with_no_compare_run(self, tmp_path: Path) -> None:
        """Clustering nothing would publish an empty analysis that looks complete."""
        with pytest.raises(FileNotFoundError, match="tehillim-compare"):
            main(["--data-root", str(tmp_path)])
