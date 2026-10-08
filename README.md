# pycicada

pycicada is a MATLAB-free implementation of the automatic subject-level CICADA
classifier. It supports comparison of CICADA component selection and
nonaggressive denoising with the ICA-AROMA workflow used by `nro`.

The implementation separates two concerns:

1. The upstream `CICADA_1_MasksandICAs.sh` stage creates anatomical noise masks,
   runs FSL MELODIC, and calculates component/region overlaps.
2. Python reconstructs CICADA's temporal and spatial features, applies its
   three-level feature clustering and signal-retention policy, and can call FSL
   `fsl_regfilt`.

This is research software. It is not yet a validated replacement for CICADA or
ICA-AROMA. The classifier is based on CICADA upstream commit
`b5a97bc3753b1cf3660c0e8add960f46e91ea95c` and the methods described in Dodd
et al., *Imaging Neuroscience* (2025), DOI `10.1162/IMAG.a.114`.

## Current commands

Run the complete automatic classifier:

```bash
pycicada run \
  --output-dir /work/cicada/sub-01/ses-01/task-rest \
  --bold /data/sub-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz \
  --mask /data/sub-01_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz \
  --confounds /data/sub-01_task-rest_desc-confounds_timeseries.tsv \
  --anat /data/sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz \
  --anat-mask /data/sub-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz \
  --gm-prob /data/sub-01_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz \
  --wm-prob /data/sub-01_space-MNI152NLin2009cAsym_label-WM_probseg.nii.gz \
  --csf-prob /data/sub-01_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz \
  --fsl-image /path/to/qunex_suite-1.5.1.sif
```

When a container is selected, input, output, and bundled-upstream directories
are bound automatically at their host paths. Extra site-specific mounts can be
added with repeated `--bind SOURCE:DESTINATION` arguments. On the Climblab
cluster, QuNex's FreeSurfer-license mount can be supplied as:

```bash
--bind /juice6/u/nlp/climblab/freesurfer/license.txt:/nro-license:ro
```

Classify an already prepared CICADA task directory:

```bash
pycicada classify \
  --task-dir /path/to/sub-01/ses-01/rest \
  --melodic-dir /path/to/sub-01/ses-01/rest/melodic
```

Add `--denoise` and supply the FSL execution options to run nonaggressive
component regression. Run `pycicada --help` for the complete interface. The
legacy `cicada-python` command remains available as an alias.

The command writes `cicada_python/feature_values.tsv`,
`cicada_python/component_labels.tsv`, component lists, and `provenance.json`.

## Scope

The first milestone covers automatic classification. Optional task-event
features have received a small exploratory real-data exercise but not
population-scale validation. Subject QC, manual relabeling, Group CICADA,
despiking, and post-denoising temporal/spatial filtering are intentionally
deferred.

For integration into NRO, read `docs/NRO_INTEGRATION_HANDOFF.md`. In particular,
NRO should own orchestration, denoising policy, and comparative QC; this package
should supply CICADA features and one-based component classifications.

## Intentional policy choices

pycicada follows CICADA's scientific decisions rather than pursuing
bitwise MATLAB equivalence. It records the following deviations in every run:

- constant or otherwise degenerate three-group features have deterministic
  behavior;
- non-finite feature values stop the run with a diagnostic;
- when CICADA's safety rule has to retain two signal components, the two
  highest-ranked components are retained (the upstream source appears to retain
  component numbers 1 and 2 despite describing the intended choice as the first
  two in the best order);
- newer FSL threshold maps containing multiple volumes are reduced to the final
  map so there remains exactly one threshold map per component.

## Licensing

This implementation follows and adapts the GPLv3 CICADA source and is therefore
licensed under GPL-3.0-or-later. See `LICENSE`.

The bundled preparation script and templates come from the pinned upstream
CICADA repository. See `THIRD_PARTY.md`.
