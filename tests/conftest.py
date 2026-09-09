"""Shared fixtures for integration tests against the compare analysis this package clusters."""

from __future__ import annotations

import os

# Limits BLAS threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tehillim_cluster.cli import method_for
from tehillim_cluster.cluster_methods import FEATURE_CLUSTERINGS
from tehillim_cluster.clustering import ClusteringResult
from tehillim_cluster.compare_dataset import compare_root, read_similarities
from tehillim_cluster.similarity import SimilarityResult

#: The published analysis this package clusters, overridable for a checkout kept elsewhere.
DATA_ROOT = Path(
    os.environ.get("TEHILLIM_DATA_DIR", Path(__file__).resolve().parents[2] / "tehillim-data")
)

#: Read at collection time, so a missing analysis skips rather than failing inside a fixture.
_SIMILARITY_TABLES = "domain=*/stage=raw/similarity.parquet"

ClusteringOf = Callable[[str], ClusteringResult]


def _unavailable() -> str | None:
    """Why the published analysis cannot be read, or None when it can."""
    root = compare_root(DATA_ROOT)
    if not root.is_dir():
        return f"no compare analysis at {root}, set TEHILLIM_DATA_DIR to a tehillim-data checkout"
    if not sorted(root.glob(_SIMILARITY_TABLES)):
        return f"no published similarity tables under {root}, run tehillim-compare first"
    return None


def _published_methods() -> list[str]:
    """Every method name the compare run published, read without building its matrices."""
    if _unavailable() is not None:
        return []
    names: set[str] = set()
    for path in sorted(compare_root(DATA_ROOT).glob(_SIMILARITY_TABLES)):
        names.update(pq.read_table(path, columns=["method"]).column("method").to_pylist())
    return sorted(names)


def _semantic_methods() -> list[str]:
    """The published methods the CLI clusters generically, which is every non-feature signal."""
    return [name for name in _published_methods() if name not in FEATURE_CLUSTERINGS]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrises any test asking for one semantic method over every published one."""
    if "semantic_method" in metafunc.fixturenames:
        metafunc.parametrize("semantic_method", _semantic_methods(), ids=str)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    reason = _unavailable()
    if reason is not None:
        skip_marker = pytest.mark.skip(reason=reason)
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_marker)

    if not os.environ.get("TEHILLIM_OPENROUTER_API_KEY"):
        api_skip_marker = pytest.mark.skip(
            reason="TEHILLIM_OPENROUTER_API_KEY is not set - skipping api_integration tests"
        )
        for item in items:
            if "api_integration" in item.keywords:
                item.add_marker(api_skip_marker)


@pytest.fixture(scope="session")
def similarities() -> dict[str, SimilarityResult]:
    """Every published similarity matrix, read once and shared across the session."""
    return {result.method: result for result in read_similarities(DATA_ROOT)}


@pytest.fixture(scope="session")
def clustering_of(similarities: dict[str, SimilarityResult]) -> ClusteringOf:
    """Partitions one published similarity the way the CLI does, memoised across the session."""
    computed: dict[str, ClusteringResult] = {}

    def compute(method: str) -> ClusteringResult:
        if method not in computed:
            similarity = similarities[method]
            computed[method] = method_for(similarity).compute(similarity)
        return computed[method]

    return compute


# --- Feature signals, one fixture per configured clustering ----------------


@pytest.fixture(scope="session")
def lexical_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The lexical partition, clustered from the published lexical similarity."""
    return clustering_of("lexical-tfidf-cosine")


@pytest.fixture(scope="session")
def root_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The consonantal-root partition."""
    return clustering_of("root-tfidf-cosine")


@pytest.fixture(scope="session")
def named_entity_identity_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The named-entity-identity partition."""
    return clustering_of("named-entity-identity-tfidf-cosine")


@pytest.fixture(scope="session")
def lexical_set_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The lexical-set partition."""
    return clustering_of("lexical-set-tfidf-cosine")


@pytest.fixture(scope="session")
def named_entity_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The named-entity-type partition."""
    return clustering_of("named-entity-tfidf-cosine")


@pytest.fixture(scope="session")
def verb_morphology_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The verb-morphology partition."""
    return clustering_of("verb-morphology-tfidf-cosine")


@pytest.fixture(scope="session")
def person_profile_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The grammatical-person partition."""
    return clustering_of("person-profile-tfidf-cosine")


@pytest.fixture(scope="session")
def clause_type_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The clause-type partition."""
    return clustering_of("clause-type-tfidf-cosine")


@pytest.fixture(scope="session")
def text_type_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The text-type partition."""
    return clustering_of("text-type-tfidf-cosine")


@pytest.fixture(scope="session")
def clause_relation_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The clause-relation partition."""
    return clustering_of("clause-relation-tfidf-cosine")


@pytest.fixture(scope="session")
def verb_sense_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """The verb-sense partition."""
    return clustering_of("verb-sense-tfidf-cosine")


# --- Semantic signals the structure assertions name individually -----------


@pytest.fixture(scope="session")
def miqrabert_mean_pool_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """MiqraBERT mean-pooled, the flat case the structure assertions pin."""
    return clustering_of("miqrabert-mean-pool-cosine")


@pytest.fixture(scope="session")
def miqrabert_soft_alignment_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """MiqraBERT soft-aligned."""
    return clustering_of("miqrabert-soft-alignment-cosine")


@pytest.fixture(scope="session")
def alephbert_mean_pool_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """AlephBERT mean-pooled, the structured case the assertions pin against MiqraBERT."""
    return clustering_of("alephbert-mean-pool-cosine")


@pytest.fixture(scope="session")
def alephbert_soft_alignment_clustering(clustering_of: ClusteringOf) -> ClusteringResult:
    """AlephBERT soft-aligned."""
    return clustering_of("alephbert-soft-alignment-cosine")
