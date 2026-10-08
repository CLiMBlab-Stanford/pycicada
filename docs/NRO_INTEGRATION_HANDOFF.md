# Agent handoff: integrating CICADA classification into NRO

## Intended boundary

Treat this repository as a component-feature and component-classification
backend. NRO should own BIDS discovery, work-directory allocation, scheduling,
registration, final component regression, derivative naming, and comparative
quality control.

The preferred backend result is:

- one-based signal and noise component identifiers;
- the feature, cluster, and ranked-label tables;
- classifier warnings; and
- preparation and classification provenance.

Do not reproduce NRO's cleaning pipeline, task modeling, connectivity analysis,
or derivative publication here. Local exploratory analyses are not part of the
published package or its API.

## Input contract

1. Allocate a unique task directory for every run. Do not execute two CICADA
   preparations concurrently in the same directory.
2. Supply BOLD, functional mask, and component spatial maps in the same voxel
   grid and affine. The Python preparation entry point now validates reused
   MELODIC geometry.
3. A reused MELODIC directory must contain `melodic_IC.nii.gz`, `melodic_mix`,
   `melodic_ICstats`, and either preassembled or `stats/` probability and
   threshold maps.
4. CICADA adds derived files to a reused MELODIC directory. NRO should therefore
   construct a run-local adapter directory containing links or copies of the
   private NRO inputs; it should not point CICADA at NRO's authoritative private
   MELODIC directory. CICADA will never recursively delete a caller-supplied
   directory, but it is not a read-only consumer.
5. Confounds must contain `framewise_displacement` and `dvars`, with one row per
   BOLD volume. Pass the repetition time explicitly when the NIfTI header is
   missing or invalid.
6. Component identifiers are one-based at the API and FSL boundaries.

## Scientific and operational warnings

- This is research software and is not a validated replacement for ICA-AROMA.
  Large-scale paired evaluation belongs in NRO and should include artifact
  reduction, signal preservation, task estimates, and connectome reliability.
- Repeatability alone is not validity: stable task-locked, physiological, or
  acquisition artifact can raise between-run agreement.
- Keep label selection separate from regression policy. NRO should compare
  CICADA and ICA-AROMA labels under matched regression as well as any native
  nonaggressive CICADA regression.
- A warning that fewer than two components passed the signal rules is a run-level
  QC failure even though the classifier retains two ranked components as a
  fail-safe.
- Decompositions with fewer than three components are rejected because they
  cannot leave both retained signal and removable noise components.
- The default smoothing-retention convention is `revised` (11 mm FWHM,
  converted to FSL sigma). Select `historical` only deliberately and preserve
  the recorded mode in provenance.
- Task-event features are optional and only exploratorily exercised. Do not make
  them an NRO default without broader validation.
- The implementation targets mathematical and policy faithfulness, not bitwise
  MATLAB equivalence. Its deliberate deviations are recorded in the output
  provenance and repository methods documentation.
- The package and adapted upstream material are GPL-3.0-or-later. Confirm that
  the way NRO installs or distributes this backend respects that license.

## Container and filesystem guidance

The CLI binds the output directory, primary inputs, bundled preparation code,
optional anatomy inputs, and a reused MELODIC directory into the selected
Singularity/Apptainer image. Site-specific resources may still require explicit
`--bind` values.

Preparation provenance records the configured container path, not the image
contents or installed FSL version. NRO should additionally record its immutable
image digest and FSL version alongside the backend result.

Network-atlas resampling is run-local and regenerated for each preparation; do
not restore the former shared cache keyed by task basename. Store large-scale
outputs in NRO work/derivative storage rather than this source checkout.

## Recommended NRO adapter sequence

1. Let NRO prepare or locate its ICA decomposition and transforms.
2. Create a unique run-local CICADA adapter directory and expose the required
   MELODIC products in the final BOLD analysis space.
3. Call `prepare_task`, then `build_features`, then `classify_components`.
4. Persist the returned tables, component IDs, warnings, and both provenance
   JSON documents.
5. Let NRO apply its selected regression implementation and publish derivatives.
6. Treat any geometry error, non-finite feature, fail-safe warning, or missing
   component family as a QC event rather than silently falling back.

Avoid importing private functions from either repository. If NRO needs a more
compact interface, add a small public adapter here and test its input/output
contract rather than coupling NRO to `analysis/c001_qc` or underscore-prefixed
helpers.
