# tehillim-cluster

## Overview

`tehillim-cluster` partitions similarity relations among the 150 Psalms that `tehillim-compare` has already published. It reconstructs one psalm-similarity matrix per upstream method, applies spectral clustering, records selection diagnostics, and compares each partition with a manually encoded index of Gunkel's classifications. Its data boundary begins with published comparison tables. Linguistic extraction and embedding aggregation belong to `tehillim-compare`.

## Data

The repository consumes `analysis=compare/stage=raw/psalms.parquet` and, in each upstream domain, `analysis=compare/domain=<domain>/stage=raw/{similarity,methods}.parquet`. The psalm table provides number, verse count, word count, and incipit. Each similarity row records a method, one unordered psalm pair, and a float64 score. The reader restores the symmetric matrix and fixes its diagonal at 1.0. `tehillim-cluster` has no direct access to BHSA, Text-Fabric, or embedding vectors.

`gunkel_genre_index.py` supplies the external comparison index. It encodes one primary category for 144 psalms in 14 genres and six chapter-level families. Psalms 27, 40, 66, 89, 96, and 144 are excluded because Gunkel classifies only parts of them or gives no whole-psalm category. Thirteen cross-listed cases and four hedged cases retain an explicit primary resolution. The index is a scholarly source rendered as a transparent data model. It is neither a gold standard nor a substitute for the history of form-critical disagreement.

The output adds assignments, two-dimensional spectral coordinates, and long-form metrics under `analysis=cluster/`. It remains conditioned by every upstream decision about witness, annotation, representation, and aggregation.

## Methodology

For each upstream matrix, the repository constructs `max(1 - similarity, 0)` as a nonnegative distance matrix and retains the source matrix as the spectral-clustering affinity matrix. The candidate range is two through ten clusters. A gap-style screen compares the observed distance distribution with 10 symmetric permutations of its off-diagonal values. A selection of one produces one cluster. Otherwise, the largest precomputed-distance silhouette selects the count. Eigengaps remain a diagnostic, not an alternative selector.

The selected partition receives a 2,000-draw label-permutation silhouette test and a 100-draw 80 percent subsampling check. The first conditions on the selected partition. The second records the frequency with which that count wins a resampled sweep. These measurements describe separation and stability in a supplied matrix. They do not establish a literary category or historical social setting.

The Gunkel comparison reports the full contingency table, purity, adjusted mutual information, adjusted Rand index, and a 2,000-draw AMI label-permutation test. Hungarian assignment supplies a one-to-one display label, while the contingency table preserves the overlap it conceals. Benjamini-Hochberg adjustment is calculated separately over the genre and family AMI test families. Upstream representation selection, cluster-count selection, and the choice among historical taxonomies remain outside that correction.

## Results

`clustering.json.baseline` is an archival 150-psalm snapshot generated on 2026-08-12. Its 79 methods predate the repository split. It lacks the source manifest, embedding identifiers, repository revisions, and cache provenance required for an audited reconstruction. It is retained for inspection, not as a current result release.

A current run emits assignments, two-dimensional spectral coordinates, selected cluster count, conditional partition p-value, subsampling stability, and genre and family alignment metrics. These outputs answer distinct questions. A matrix may show nonuniform structure. A selected partition may recur under subsampling. A partition may agree with the encoded Gunkel index. None of these observations entails either of the others.

The archived record contains two safeguards against overstatement. The text-type matrix received a gap-screen selection of one cluster although the silhouette sweep alone selected eight. A lexical partition could also show detectable agreement with Gunkel while grouping shared vocabulary. The first result identifies a selection failure mode. The second identifies thematic confounding. Both limit the claim that any cluster recovers genre.

## Limitations

This repository clusters relationships supplied by `tehillim-compare`. It cannot inspect the words, clause structure, half-verse boundaries, vectors, or transformations that produced a score. A cluster can therefore reflect an upstream feature distribution, aggregation rule, or embedding geometry. The clustering step contributes no independent linguistic validation.

Spectral clustering depends on an affinity representation and a count-selection rule. The gap screen, silhouette, eigengap, permutation statistic, and resampling check can diverge. The partition p-value is conditional on a selected partition and leaves the full analytic search untested.

Gunkel's categories include mixed and partial cases, and the primary index resolves some source ambiguity. Agreement measures consistency with that resolution. It cannot validate the historical categories, prove genre membership, or constitute a form-critical interpretation. The index covers 144 psalms, and the same Psalter supplies representation, partition, and comparison. There is no independent confirmation set.

## Reproducibility

The package requires Python 3.12 or later with NumPy, SciPy, scikit-learn, and PyArrow. Reproduction requires the exact compare Parquet tree, method descriptions, repository revision, command options, and hashes of the cluster tables. BHSA and embedding files are unnecessary at this stage, although their versions remain essential upstream provenance.

Set `TEHILLIM_DATA_DIR` or pass `--data-root` to the shared data root. The caches fingerprint matrix and result content but do not retain every source revision or the complete selection configuration. Clear them before an audited run and archive input and output hashes together. `./check.sh` performs formatting, static analysis, dependency checks, dead-code analysis, and unit tests. It repeats the software procedure, not the scholarly decisions encoded upstream or in the Gunkel index.

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Usage

```bash
.venv/bin/tehillim-cluster \
  --data-root /path/to/tehillim-data

./check.sh
```

Run `tehillim-compare` first against the same data root. The command writes cluster tables under `/path/to/tehillim-data/analysis=cluster/`, a clustering payload under `analysis=cluster/stage=ui/`, and the Gunkel reference payload under `reference/stage=ui/`.

## References

Gunkel, Hermann, and Joachim Begrich. *Einleitung in die Psalmen: Die Gattungen der religiösen Lyrik Israels*. Zu Ende geführt von Joachim Begrich. Göttingen: Vandenhoeck & Ruprecht, 1933.

Hubert, Lawrence, and Phipps Arabie. [“Comparing Partitions.”](https://doi.org/10.1007/BF01908075) *Journal of Classification* 2, no. 1 (1985): 193–218.

Kuhn, Harold W. [“The Hungarian Method for the Assignment Problem.”](https://doi.org/10.1002/nav.3800020109) *Naval Research Logistics Quarterly* 2, nos. 1–2 (1955): 83–97.

Ng, Andrew Y., Michael I. Jordan, and Yair Weiss. [“On Spectral Clustering: Analysis and an Algorithm.”](https://ai.stanford.edu/~ang/papers/nips01-spectral.pdf) In *Advances in Neural Information Processing Systems 14*, 849–856, 2002.

Rousseeuw, Peter J. [“Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.”](https://doi.org/10.1016/0377-0427(87)90125-7) *Journal of Computational and Applied Mathematics* 20 (1987): 53–65.

Tibshirani, Robert, Guenther Walther, and Trevor Hastie. [“Estimating the Number of Clusters in a Data Set via the Gap Statistic.”](https://doi.org/10.1111/1467-9868.00293) *Journal of the Royal Statistical Society: Series B* 63, no. 2 (2001): 411–423.

Vinh, Nguyen Xuan, Julien Epps, and James Bailey. [“Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance.”](https://jmlr.org/papers/v11/vinh10a.html) *Journal of Machine Learning Research* 11 (2010): 2837–2854.

## License

The source code is released under the MIT License. The input comparison tables and historical reference materials retain their own provenance, licences, and access conditions.
