# Prototype methods and fidelity notes

## Scientific target

pycicada targets the automatic subject-level CICADA policy described by
Dodd et al. (2025) and encoded in upstream commit
`b5a97bc3753b1cf3660c0e8add960f46e91ea95c`. It is intended as an empirical
alternative to ICA-AROMA, not as a binary-compatible MATLAB replacement.

## Processing graph

1. Resample anatomical masks to the BOLD grid and construct CICADA's gray
   matter, white matter, CSF, edge, susceptibility, subependymal, in-brain, and
   out-of-brain probability regions.
2. Run FSL MELODIC with `--Ostats --nobet --mmthresh=0.5` and the BOLD TR.
3. Calculate absolute spatial-map mass inside each anatomical region and seven
   functional-network atlas regions.
4. Derive the component features below.
5. Split each feature into low, middle, and high groups using one-dimensional
   Lloyd clustering initialized at its minimum, median, and maximum.
6. Rank components by range-normalized smoothing retention × squared gray
   matter overlap × HRF power overlap.
7. Traverse the above-mean part of this ranking. Apply CICADA's signal/noise
   evidence rules and stop once the configurable noise tolerance is exhausted.
8. Regress selected noise components nonaggressively with FSL `fsl_regfilt`.

## Feature families

### Spatial anatomy

The component mass in a region is the FSL nonzero mean multiplied by the
nonzero voxel count. Gray matter, white matter, CSF, edge, susceptibility,
subependymal, and out-of-brain features retain CICADA's ratios. In particular,
noise-region mass is evaluated relative to gray-matter mass and then divided by
gray-matter proportion again. This deliberately weights gray-matter overlap
more heavily than a simple regional fraction would.

### Spatial networks

Absolute component maps are intersected with CICADA's seven-network atlas.
Network overlap is retained primarily as interpretable supporting evidence; it
does not override the main ranking and decision policy.

### Smoothness retention

The feature is the total absolute spatial-map mass after 11 mm FWHM Gaussian
smoothing divided by mass before smoothing. FSL receives sigma in millimeters,
so the preparation stage uses `11 / 2.354820045` mm.

### Frequency and HRF similarity

Component spectra follow the upstream FFT and trapezoidal-integration
construction. Power is divided into less than approximately 0.008 Hz, the
0.008–0.15 Hz BOLD band, and frequencies above approximately 0.15 Hz. The
general expected response is CICADA's sampled SPM-style double-gamma HRF.
Similarity features are its spectral dot product with each component and their
squared spectral distance.

For task data, BIDS onset/duration blocks are convolved with the same HRF.
Each component is paired with its most-correlated condition spectrum. Task
features are optional and remain secondary evidence, as in upstream CICADA.

### Motion and transient noise

Squared correlations relate detrended framewise displacement and DVARS to the
absolute first difference of each detrended component time series. Spikiness is
the maximum absolute component time-series value; it becomes adverse evidence
only above the upstream absolute threshold of 5.

## Conservative signal policy

The classifier is intentionally conservative about deleting neuronal signal.
Strong gray-matter overlap combined with either HRF spectral support or
smoothness is sufficient signal evidence. Components without strong gray
matter evidence must also lack high anatomical noise-region evidence. Low gray
matter or low best HRF overlap is disqualifying, while motion, DVARS,
high-frequency, edge, CSF, susceptibility, and out-of-brain tags contribute
noise evidence.

The traversal tolerance rises after a retained signal component and falls
after a rejected component, capped at its starting value. Components below the
above-mean section of the ranking are noise. If fewer than two components are
retained, pycicada preserves the two highest-ranked components and emits a
QC-failure warning.

## Deliberate engineering deviations

- Degenerate one-dimensional clustering is deterministic. A constant feature
  is neither high nor low.
- Invalid or non-finite feature inputs fail explicitly.
- Component identifiers remain one-based at FSL and user-facing boundaries.
- FSL 6.0.7 can emit multi-volume `thresh_zstat` files for individual
  components. The final volume is retained, yielding exactly one map per
  component, following the handling already used by `nro` for ICA-AROMA.
- State is stored as TSV and JSON rather than opaque MATLAB structures.

Every result records these policies, the pycicada version, and the targeted
upstream commit in `provenance.json`. Preparation additionally records the
modified-script digest, smoothing-retention mode, generated/reused MELODIC
mode, component-map source, container image, and input paths in
`cicada_python/preparation.json`; classification provenance embeds that record
when it is available.
