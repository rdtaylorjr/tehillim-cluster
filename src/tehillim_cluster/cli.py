"""Command-line entrypoint: partition the psalms from the published compare analysis."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from tehillim_cluster.atomic_write import write_json
from tehillim_cluster.build_cache import (
    fingerprint_similarity,
    load_cached_clustering,
    write_cached_clustering,
)
from tehillim_cluster.cluster_methods import FEATURE_CLUSTERINGS
from tehillim_cluster.compare_dataset import (
    ANALYSIS_CLUSTER,
    REFERENCE_ROOT,
    read_psalms,
    read_similarities,
)
from tehillim_cluster.dataset import ui_path, write_cluster_dataset
from tehillim_cluster.embedding import compute_embedding
from tehillim_cluster.export_clustering import build_clustering_payload
from tehillim_cluster.export_gunkel import build_gunkel_payload
from tehillim_cluster.semantic_clustering import clustering_for

if TYPE_CHECKING:
    from tehillim_cluster.clustering import ClusteringMethod, ClusteringResult
    from tehillim_cluster.similarity import SimilarityResult

#: repo_root/data/.
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

#: Root of the partitioned dataset this package reads from and writes into.
DEFAULT_DATA_ROOT = _DATA_DIR

#: repo_root/data/clustering_cache.
DEFAULT_CLUSTERING_CACHE_DIR = _DATA_DIR / "clustering_cache"

#: The clustering shown on page load.
DEFAULT_CLUSTER_METHOD = "alephbert-mean-pool-spectral"


def default_cluster_output(data_root: Path) -> Path:
    """Where the cluster payload lands when no explicit path is given."""
    return ui_path(data_root, ANALYSIS_CLUSTER, "clustering.json")


def default_gunkel_output(data_root: Path) -> Path:
    """Where the Gunkel reference lands: neutral, because both analyses' pages read it."""
    return Path(data_root / REFERENCE_ROOT / "stage=ui" / "gunkel.json")


def parse_args(
    argv: list[str] | None = None, env: Mapping[str, str] | None = None
) -> argparse.Namespace:
    """Parses the arguments this module documents."""
    environment: Mapping[str, str] = os.environ if env is None else env
    parser = argparse.ArgumentParser(description=__doc__)
    env_data_root = environment.get("TEHILLIM_DATA_DIR")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(env_data_root) if env_data_root else DEFAULT_DATA_ROOT,
        help=(
            "Root of the partitioned dataset, read at analysis=compare and written at "
            f"analysis=cluster (default: $TEHILLIM_DATA_DIR, else {DEFAULT_DATA_ROOT})"
        ),
    )
    parser.add_argument(
        "--cluster-output",
        type=Path,
        default=None,
        help="Cluster payload JSON path (default: under --data-root at analysis=cluster/stage=ui)",
    )
    parser.add_argument(
        "--gunkel-output",
        type=Path,
        default=None,
        help="Gunkel reference JSON path (default: under --data-root at analysis=cluster/stage=ui)",
    )
    env_cache_dir = environment.get("TEHILLIM_CLUSTERING_CACHE_DIR")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(env_cache_dir) if env_cache_dir else DEFAULT_CLUSTERING_CACHE_DIR,
        help=f"Directory for cached clustering results (default: {DEFAULT_CLUSTERING_CACHE_DIR})",
    )
    return parser.parse_args(argv)


def _write_payload(path: Path, payload: object) -> None:
    """Writes one site payload atomically and reports the size it published."""
    write_json(path, payload, compact=True)
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KiB)", file=sys.stderr)


def _cluster_once(
    cache_dir: Path, method: ClusteringMethod, similarity: SimilarityResult
) -> ClusteringResult:
    """One method's partition, reused only when the cached entry came from this similarity."""
    fingerprint = fingerprint_similarity(similarity)
    cached = load_cached_clustering(cache_dir, method.name, fingerprint=fingerprint)
    if cached is not None:
        print(f"Clustering {method.name}... (cached)", file=sys.stderr)
        return cached
    print(f"Clustering {method.name}...", file=sys.stderr)
    result = method.compute(similarity)
    write_cached_clustering(cache_dir, result, fingerprint=fingerprint)
    return result


@dataclass(frozen=True, slots=True)
class _Described:
    """Adapts a stored result to the name-and-prose shape a derived clustering needs."""

    name: str
    description: str


def method_for(similarity: SimilarityResult) -> ClusteringMethod:
    """The clustering configured for one similarity, whether a feature signal or a semantic one."""
    configured = FEATURE_CLUSTERINGS.get(similarity.method)
    if configured is not None:
        return configured
    return clustering_for(_Described(similarity.method, similarity.description))


def run(
    data_root: Path,
    cluster_output: Path | None = None,
    gunkel_output: Path | None = None,
    cache_dir: Path = DEFAULT_CLUSTERING_CACHE_DIR,
) -> None:
    """Partitions every similarity the compare analysis published."""
    cluster_output = cluster_output or default_cluster_output(data_root)
    gunkel_output = gunkel_output or default_gunkel_output(data_root)

    print("Reading the compare analysis...", file=sys.stderr)
    psalms = read_psalms(data_root)
    similarities = read_similarities(data_root)
    print(f"  {len(psalms)} psalms, {len(similarities)} similarity signals", file=sys.stderr)

    results = [_cluster_once(cache_dir, method_for(s), s) for s in similarities]
    embeddings = [compute_embedding(s) for s in similarities]

    print("Building clustering payload...", file=sys.stderr)
    payload = build_clustering_payload(
        psalms=psalms,
        results=results,
        embeddings=embeddings,
        default_method=DEFAULT_CLUSTER_METHOD,
        cache_dir=cache_dir,
    )
    _write_payload(cluster_output, payload)

    print("Building Gunkel reference payload...", file=sys.stderr)
    _write_payload(gunkel_output, build_gunkel_payload())

    print("Writing the partitioned dataset...", file=sys.stderr)
    for path in write_cluster_dataset(
        data_root,
        clusterings=results,
        embeddings=embeddings,
        cluster_methods=payload["clusterMethods"],
    ):
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KiB)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Entry point: parses the command line and runs the clustering."""
    args = parse_args(argv)
    run(args.data_root, args.cluster_output, args.gunkel_output, args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
