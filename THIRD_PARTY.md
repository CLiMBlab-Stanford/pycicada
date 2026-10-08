# Third-party material

## CICADA

- Source: <https://github.com/keithcdodd/CICADA>
- Pinned source commit: `b5a97bc3753b1cf3660c0e8add960f46e91ea95c`
- License: GNU General Public License version 3
- Bundled paths:
  `src/cicada_python/vendor/CICADA/basescripts/CICADA_1_MasksandICAs.sh` and
  `src/cicada_python/vendor/CICADA/templates/`

The preparation script contains these local adaptations:

- normalize multi-volume MELODIC `thresh_zstat` products to one final volume
  per component, matching the scientific expectation of one spatial map per
  independent component;
- accept preassembled probability and threshold maps so an NRO decomposition
  transformed into the analysis space can be reused;
- validate caller-supplied MELODIC directories and fail on an incomplete one
  instead of deleting and rebuilding it; and
- resample the functional-network atlas independently inside each run directory
  instead of sharing a task-name-keyed cache across runs.

The exact modified-script digest, selected smoothing-retention mode, and
preparation mode are written to each run's `cicada_python/preparation.json`.
