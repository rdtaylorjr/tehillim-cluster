"""Serializes psalms and clustering results into clustering.json, the Cluster page's payload."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tehillim_cluster.analysis import benjamini_hochberg
from tehillim_cluster.build_cache import (
    input_fingerprint,
    load_cached_cluster_payload,
    write_cached_cluster_payload,
)
from tehillim_cluster.clustering import ClusteringResult
from tehillim_cluster.compare_dataset import CORPUS_SOURCE, psalm_core
from tehillim_cluster.compare_dataset import PsalmFacts as Psalm
from tehillim_cluster.embedding import Embedding2D
from tehillim_cluster.genre_alignment import (
    GenreAlignment,
    compute_family_alignment,
    compute_genre_alignment,
)
from tehillim_cluster.k_selection import DEFAULT_N_PERMUTATIONS


def build_clustering_payload(
    psalms: list[Psalm],
    results: list[ClusteringResult],
    embeddings: list[Embedding2D],
    default_method: str,
    *,
    ami_permutations: int = DEFAULT_N_PERMUTATIONS,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Assembles the Cluster page's payload: per-signal AMI-permutation tests."""
    cluster_methods = [
        _cluster_method_payload(
            result, embedding, ami_permutations=ami_permutations, cache_dir=cache_dir
        )
        for result, embedding in zip(results, embeddings, strict=True)
    ]
    _attach_adjusted_ami_p_values(cluster_methods, alignment_key="genreAlignment")
    _attach_adjusted_ami_p_values(cluster_methods, alignment_key="familyAlignment")

    return {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": CORPUS_SOURCE,
        "psalms": [psalm_core(psalm) for psalm in psalms],
        "clusterMethods": cluster_methods,
        "defaultClusterMethod": default_method,
    }


def _attach_adjusted_ami_p_values(
    cluster_methods: list[dict[str, Any]], *, alignment_key: str
) -> None:
    raw_p_values = [method[alignment_key]["amiPValue"] for method in cluster_methods]
    adjusted = benjamini_hochberg(raw_p_values)
    for method, adjusted_p_value in zip(cluster_methods, adjusted, strict=True):
        method[alignment_key]["amiPValueAdjusted"] = adjusted_p_value


def _cluster_method_payload(
    result: ClusteringResult,
    embedding: Embedding2D,
    *,
    ami_permutations: int = DEFAULT_N_PERMUTATIONS,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    fingerprint = input_fingerprint(
        list(result.psalm_numbers),
        list(result.labels),
        result.n_clusters,
        result.partition_p_value,
        result.k_stability,
        list(embedding.x),
        list(embedding.y),
        ami_permutations,
    )
    if cache_dir is not None:
        cached = load_cached_cluster_payload(cache_dir, result.method, fingerprint=fingerprint)
        if cached is not None:
            return cached

    if embedding.psalm_numbers != result.psalm_numbers:
        raise ValueError(
            f"embedding psalm_numbers for {embedding.method!r} don't match "
            f"clustering psalm_numbers for {result.method!r}"
        )
    assignments = {
        str(psalm_number): label
        for psalm_number, label in zip(result.psalm_numbers, result.labels, strict=True)
    }

    members: dict[int, list[int]] = {index: [] for index in range(result.n_clusters)}
    for psalm_number, label in zip(result.psalm_numbers, result.labels, strict=True):
        members[label].append(psalm_number)

    clusters = [
        {"index": index, "size": len(psalm_numbers), "psalmNumbers": psalm_numbers}
        for index, psalm_numbers in sorted(members.items())
    ]

    payload = {
        "id": result.method,
        "description": result.description,
        "nClusters": result.n_clusters,
        "partitionPValue": result.partition_p_value,
        "kStability": result.k_stability,
        "assignments": assignments,
        "clusters": clusters,
        "embedding": {
            "x": list(embedding.x),
            "y": list(embedding.y),
            "structureCaptured": embedding.structure_captured,
        },
        "genreAlignment": _alignment_payload(
            compute_genre_alignment(result, ami_permutations=ami_permutations)
        ),
        "familyAlignment": _alignment_payload(
            compute_family_alignment(result, ami_permutations=ami_permutations)
        ),
    }
    if cache_dir is not None:
        write_cached_cluster_payload(cache_dir, result.method, payload, fingerprint=fingerprint)
    return payload


def _alignment_payload(alignment: GenreAlignment) -> dict[str, Any]:
    return {
        "genres": list(alignment.genres),
        "counts": [list(row) for row in alignment.counts],
        "genreTotals": list(alignment.genre_totals),
        "clusterGenreLabels": list(alignment.cluster_genre_labels),
        "purity": alignment.purity,
        "ami": alignment.ami,
        "ari": alignment.ari,
        "amiPValue": alignment.ami_p_value,
    }
