"""Configured clustering methods: one `SpectralClusteringMethod` per shipped similarity signal."""

from __future__ import annotations

from tehillim_cluster.clustering import SpectralClusteringMethod, data_driven_k
from tehillim_cluster.k_selection import DEFAULT_K_VALUES

#: Search range for the data-driven k-selector below.
_K_VALUES = DEFAULT_K_VALUES

# --- Lexical / vocabulary-based: thematic clusters, not genre --------------

LEXICAL_CLUSTERING = SpectralClusteringMethod(
    name="lexical-spectral",
    description=(
        "Spectral clustering over lexical similarity. Partitions the "
        "Psalter by shared Biblical Hebrew content-word vocabulary (nouns, "
        "verbs, adjectives, adverbs, proper nouns, interjections). A "
        "thematic grouping (which words two psalms share), not a genre "
        "grouping. Contrast with the syntactic/clause-structure methods "
        "below, which cluster on grammatical form instead."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

ROOT_CLUSTERING = SpectralClusteringMethod(
    name="root-spectral",
    description=(
        "Spectral clustering over root similarity. Partitions the Psalter "
        "by shared triliteral consonantal roots, a coarser cousin of "
        "lexical clustering that credits shared thematic vocabulary across "
        "derivationally related words (e.g. a verb and its cognate noun) "
        "that lexical similarity keeps distinct. Thematic, not genre."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

NAMED_ENTITY_IDENTITY_CLUSTERING = SpectralClusteringMethod(
    name="named-entity-identity-spectral",
    description=(
        "Spectral clustering over named-entity-identity similarity. "
        "Partitions the Psalter by which specific proper nouns (personal "
        "names, place names, ...) two psalms share, e.g. a Zion-naming "
        "cluster distinct from a Sinai-naming one. Thematic, not genre."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

LEXICAL_SET_CLUSTERING = SpectralClusteringMethod(
    name="lexical-set-spectral",
    description=(
        "Spectral clustering over lexical-set similarity. Partitions the "
        "Psalter by numeral, focus-particle, and grammaticalized-"
        "preposition/adverb/copula tag frequency profile, a finer "
        "part-of-speech subclassification than `sp`. Sits between the "
        "lexical and syntactic families: closer to vocabulary than to "
        "clause structure, so read its clusters as thematic leanings, not "
        "genre."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

NAMED_ENTITY_CLUSTERING = SpectralClusteringMethod(
    name="named-entity-spectral",
    description=(
        "Spectral clustering over named-entity-type similarity. "
        "Partitions the Psalter by onomastic register (person-name-dense "
        "vs. place-name-dense vs. deity-name-dense, ...), independent of "
        "which specific names appear. Thematic, not genre."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

# --- Syntactic / grammatical-profile: candidate form-critical signals -----

VERB_MORPHOLOGY_CLUSTERING = SpectralClusteringMethod(
    name="verb-morphology-spectral",
    description=(
        "Spectral clustering over verb-morphology similarity. Partitions "
        "the Psalter by verb stem/conjugation tag profile (including "
        "participles), motivated by Gunkel's form-critical genre "
        "categories but validated only narrowly against them. See the "
        "README."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

PERSON_PROFILE_CLUSTERING = SpectralClusteringMethod(
    name="person-profile-spectral",
    description=(
        "Spectral clustering over grammatical-person similarity. "
        "Partitions the Psalter by individual vs. communal address, a "
        "classical form-critical marker distinct from verb morphology."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

CLAUSE_TYPE_CLUSTERING = SpectralClusteringMethod(
    name="clause-type-spectral",
    description=(
        "Spectral clustering over clause-type similarity. Partitions the "
        "Psalter by constituent-order/verb-form clause pattern, the most "
        "discriminative signal of any tried (66.7% of pairs score below "
        "0.5)."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

TEXT_TYPE_CLUSTERING = SpectralClusteringMethod(
    name="text-type-spectral",
    description=(
        "Spectral clustering over text-type similarity. Partitions the "
        "Psalter by narrative/discursive/quotation discourse register, "
        "BHSA's closest analogue to a discourse-register feature."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

CLAUSE_RELATION_CLUSTERING = SpectralClusteringMethod(
    name="clause-relation-spectral",
    description=(
        "Spectral clustering over clause-relation similarity. Partitions "
        "the Psalter by how clauses relate to their context (coordinated, "
        "attributive, object clause, ...). Sparse (22.5% of words) but "
        "real signal."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

VERB_SENSE_CLUSTERING = SpectralClusteringMethod(
    name="verb-sense-spectral",
    description=(
        "Spectral clustering over verb-sense similarity. Partitions the "
        "Psalter by verb argument-realization pattern (ETCBC/valence), "
        "covering 60.6% of Psalter verb occurrences."
    ),
    k_selector=data_driven_k(_K_VALUES),
)

#: The clustering configured for each feature signal, keyed by the similarity it partitions.
FEATURE_CLUSTERINGS: dict[str, SpectralClusteringMethod] = {
    "lexical-tfidf-cosine": LEXICAL_CLUSTERING,
    "root-tfidf-cosine": ROOT_CLUSTERING,
    "named-entity-identity-tfidf-cosine": NAMED_ENTITY_IDENTITY_CLUSTERING,
    "verb-morphology-tfidf-cosine": VERB_MORPHOLOGY_CLUSTERING,
    "person-profile-tfidf-cosine": PERSON_PROFILE_CLUSTERING,
    "lexical-set-tfidf-cosine": LEXICAL_SET_CLUSTERING,
    "named-entity-tfidf-cosine": NAMED_ENTITY_CLUSTERING,
    "clause-type-tfidf-cosine": CLAUSE_TYPE_CLUSTERING,
    "text-type-tfidf-cosine": TEXT_TYPE_CLUSTERING,
    "clause-relation-tfidf-cosine": CLAUSE_RELATION_CLUSTERING,
    "verb-sense-tfidf-cosine": VERB_SENSE_CLUSTERING,
}
