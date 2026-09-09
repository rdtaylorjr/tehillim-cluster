"""Symbols vulture cannot see used: result-record fields read only after serialization."""

from tehillim import analysis, gunkel_genre_index

#: Reported in the analysis payload rather than read in process, so no call site references them.
analysis.PermutationResult.null_mean
analysis.PermutationResult.null_std
gunkel_genre_index.GunkelClassification.subtype
gunkel_genre_index.GunkelClassification.citation
