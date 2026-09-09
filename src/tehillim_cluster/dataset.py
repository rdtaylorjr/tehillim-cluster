"""Writes compare and cluster results as partitioned Parquet, in the benchmarks tree's grammar."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tehillim_cluster.cluster_methods import FEATURE_CLUSTERINGS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from tehillim_cluster.clustering import ClusteringResult
    from tehillim_cluster.embedding import Embedding2D

ANALYSIS_COMPARE = "compare"
ANALYSIS_CLUSTER = "cluster"

#: The domain each non-semantic method's features come from, in the vocabulary the other repos use.
_FEATURE_METHOD_DOMAINS = {
    "lexical-tfidf-cosine": "lexical",
    "root-tfidf-cosine": "lexical",
    "lexical-set-tfidf-cosine": "lexical",
    "named-entity-tfidf-cosine": "lexical",
    "named-entity-identity-tfidf-cosine": "lexical",
    #: BHSA's sense is a lexeme-level annotation, where domain=semantic is embedding models only.
    "verb-sense-tfidf-cosine": "lexical",
    "verb-morphology-tfidf-cosine": "morphological",
    "person-profile-tfidf-cosine": "morphological",
    "clause-type-tfidf-cosine": "syntactic",
    "text-type-tfidf-cosine": "syntactic",
    "clause-relation-tfidf-cosine": "syntactic",
}


#: Each feature clustering's domain, composed from the similarity it partitions.
_FEATURE_CLUSTERING_DOMAINS = {
    clustering.name: _FEATURE_METHOD_DOMAINS[similarity_name]
    for similarity_name, clustering in FEATURE_CLUSTERINGS.items()
}


def partition_path(data_root: Path, analysis: str, domain: str, stage: str, name: str) -> Path:
    """The Hive-partitioned location of one table, matching the benchmarks tree's dimensions."""
    return data_root / f"analysis={analysis}" / f"domain={domain}" / f"stage={stage}" / name


def ui_path(data_root: Path, analysis: str, name: str) -> Path:
    """Where an analysis's UI payload lands, with no domain level since it spans every domain."""
    return data_root / f"analysis={analysis}" / "stage=ui" / name


def domain_of(method_name: str) -> str:
    """Which domain a method scores, accepting either the similarity name or its partition's."""
    for table in (_FEATURE_CLUSTERING_DOMAINS, _FEATURE_METHOD_DOMAINS):
        domain = table.get(method_name)
        if domain is not None:
            return domain
    if method_name.endswith(("-spectral", "-cosine")):
        #: Every remaining signal in this analysis comes from a semantic embedding.
        return "semantic"
    raise KeyError(
        f"{method_name} has no domain: add it to _FEATURE_METHOD_DOMAINS or name it -spectral"
    )


def assignments_table(results: Sequence[ClusteringResult]) -> pa.Table:
    """Which cluster each psalm landed in, for each method."""
    methods: list[str] = []
    psalms: list[np.ndarray] = []
    clusters: list[np.ndarray] = []
    for result in results:
        psalms.append(np.asarray(result.psalm_numbers))
        clusters.append(np.asarray(result.labels))
        methods.extend([result.method] * len(result.psalm_numbers))
    return pa.table(
        {
            "method": pa.array(methods, type=pa.large_string()),
            "psalm": pa.array(_concatenate(psalms), type=pa.int32()),
            "cluster": pa.array(_concatenate(clusters), type=pa.int32()),
        }
    )


def embedding_table(embeddings: Sequence[Embedding2D]) -> pa.Table:
    """Each psalm's two-dimensional position, for each method."""
    methods: list[str] = []
    psalms: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for embedding in embeddings:
        psalms.append(np.asarray(embedding.psalm_numbers))
        xs.append(np.asarray(embedding.x, dtype=np.float64))
        ys.append(np.asarray(embedding.y, dtype=np.float64))
        methods.extend([embedding.method] * len(embedding.psalm_numbers))
    return pa.table(
        {
            "method": pa.array(methods, type=pa.large_string()),
            "psalm": pa.array(_concatenate(psalms), type=pa.int32()),
            "x": pa.array(_concatenate(xs), type=pa.float64()),
            "y": pa.array(_concatenate(ys), type=pa.float64()),
        }
    )


def _concatenate(parts: Sequence[np.ndarray]) -> np.ndarray:
    """One array from many, staying in numpy rather than building a Python list per element."""
    if not parts:
        return np.empty(0)
    return np.concatenate(parts)


def write_table(table: pa.Table, path: Path) -> None:
    """Writes one table to its partition, creating the partition directories it needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


_CLUSTER_METRICS: tuple[tuple[str, tuple[str, ...], tuple[str, ...] | None], ...] = (
    ("n_clusters", ("nClusters",), None),
    ("partition_p_value", ("partitionPValue",), None),
    ("k_stability", ("kStability",), None),
    ("structure_captured", ("embedding", "structureCaptured"), None),
    ("genre_purity", ("genreAlignment", "purity"), None),
    ("genre_ami", ("genreAlignment", "ami"), None),
    ("genre_ari", ("genreAlignment", "ari"), None),
    (
        "genre_ami_p_value",
        ("genreAlignment", "amiPValue"),
        ("genreAlignment", "amiPValueAdjusted"),
    ),
    ("family_purity", ("familyAlignment", "purity"), None),
    ("family_ami", ("familyAlignment", "ami"), None),
    ("family_ari", ("familyAlignment", "ari"), None),
    (
        "family_ami_p_value",
        ("familyAlignment", "amiPValue"),
        ("familyAlignment", "amiPValueAdjusted"),
    ),
)


_CORRECTED_ACROSS = "method"


def _dig(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:  # noqa: ANN401
    """Follows a key path into the export payload."""
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def cluster_metrics_table(cluster_methods: Sequence[Mapping[str, Any]]) -> pa.Table:
    """One row per method per scalar, in the long schema the benchmarks master tables use."""
    methods: list[str] = []
    domains: list[str] = []
    metrics: list[str] = []
    values: list[float | None] = []
    q_values: list[float | None] = []
    q_value_by: list[str | None] = []
    for method in cluster_methods:
        name = method["id"]
        domain = domain_of(name)
        for metric, path, adjusted_path in _CLUSTER_METRICS:
            raw = _dig(method, path)
            methods.append(name)
            domains.append(domain)
            metrics.append(metric)
            values.append(None if raw is None else float(raw))
            adjusted = None if adjusted_path is None else _dig(method, adjusted_path)
            q_values.append(None if adjusted is None else float(adjusted))
            q_value_by.append(None if adjusted_path is None else _CORRECTED_ACROSS)
    return pa.table(
        {
            "method": pa.array(methods, type=pa.large_string()),
            "domain": pa.array(domains, type=pa.large_string()),
            "metric": pa.array(metrics, type=pa.large_string()),
            "value": pa.array(values, type=pa.float64()),
            "q_value": pa.array(q_values, type=pa.float64()),
            "q_value_by": pa.array(q_value_by, type=pa.large_string()),
        }
    )


def write_cluster_dataset(
    data_root: Path,
    *,
    clusterings: Sequence[ClusteringResult],
    embeddings: Sequence[Embedding2D],
    cluster_methods: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Writes every cluster table, one partition per domain, and reports what it wrote."""
    written: list[Path] = []
    for domain in sorted({domain_of(result.method) for result in clusterings}):
        partitions = (
            (
                "profiles",
                "assignments.parquet",
                assignments_table([r for r in clusterings if domain_of(r.method) == domain]),
            ),
            (
                "detail",
                "embedding.parquet",
                embedding_table([e for e in embeddings if domain_of(e.method) == domain]),
            ),
            (
                "master",
                "method_metrics_long.parquet",
                cluster_metrics_table([m for m in cluster_methods if domain_of(m["id"]) == domain]),
            ),
        )
        for stage, name, table in partitions:
            path = partition_path(data_root, ANALYSIS_CLUSTER, domain, stage, name)
            write_table(table, path)
            written.append(path)
    return written
