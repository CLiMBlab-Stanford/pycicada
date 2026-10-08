"""Invoke CICADA's upstream FSL preparation stage without MATLAB."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from .fsl import FSLRunner

UPSTREAM_COMMIT = "b5a97bc3753b1cf3660c0e8add960f46e91ea95c"
SMOOTHING_RETENTION_MODES = ("revised", "historical")


def bundled_upstream_dir() -> Path:
    """Return the package's pinned CICADA FSL-stage directory."""
    return Path(__file__).resolve().parent / "vendor" / "CICADA"


def _validate_reusable_melodic_dir(melodic_dir: Path, functional_mask: Path) -> dict[str, str]:
    """Reject incomplete or spatially incompatible reused decompositions."""
    if not melodic_dir.is_dir():
        raise FileNotFoundError(f"Reused MELODIC directory does not exist: {melodic_dir}")
    required = ("melodic_IC.nii.gz", "melodic_mix", "melodic_ICstats")
    missing = [name for name in required if not (melodic_dir / name).is_file()]
    has_probability_maps = (melodic_dir / "ICprobabilities.nii.gz").is_file() or any(
        (melodic_dir / "stats").glob("probmap_*")
    )
    has_threshold_maps = (melodic_dir / "ICthresh_zstat.nii.gz").is_file() or any(
        (melodic_dir / "stats").glob("thresh_zstat*")
    )
    if not has_probability_maps:
        missing.append("ICprobabilities.nii.gz or stats/probmap_*")
    if not has_threshold_maps:
        missing.append("ICthresh_zstat.nii.gz or stats/thresh_zstat*")
    if missing:
        raise ValueError(
            f"Refusing to use incomplete reused MELODIC directory {melodic_dir}; "
            "missing: " + ", ".join(missing)
        )

    probability_source = (
        melodic_dir / "ICprobabilities.nii.gz"
        if (melodic_dir / "ICprobabilities.nii.gz").is_file()
        else next((melodic_dir / "stats").glob("probmap_*"))
    )
    threshold_source = (
        melodic_dir / "ICthresh_zstat.nii.gz"
        if (melodic_dir / "ICthresh_zstat.nii.gz").is_file()
        else next((melodic_dir / "stats").glob("thresh_zstat*"))
    )
    reference = nib.load(str(functional_mask))
    component_image = nib.load(str(melodic_dir / "melodic_IC.nii.gz"))
    component_count = component_image.shape[3] if len(component_image.shape) == 4 else 1
    for label, path in (
        ("melodic_IC", melodic_dir / "melodic_IC.nii.gz"),
        ("probability map", probability_source),
        ("threshold map", threshold_source),
    ):
        image = nib.load(str(path))
        if image.shape[:3] != reference.shape[:3] or not np.allclose(
            image.affine, reference.affine, rtol=1e-5, atol=1e-4
        ):
            raise ValueError(
                f"Reused MELODIC {label} geometry does not match the functional mask: {path}"
            )
    for label, source in (("probability", probability_source), ("threshold", threshold_source)):
        if source.parent == melodic_dir:
            image = nib.load(str(source))
            volume_count = image.shape[3] if len(image.shape) == 4 else 1
        else:
            pattern = "probmap_*" if label == "probability" else "thresh_zstat*"
            volume_count = sum(1 for _ in source.parent.glob(pattern))
        if volume_count != component_count:
            raise ValueError(
                f"Reused MELODIC {label} map count ({volume_count}) does not match "
                f"component count ({component_count})"
            )
    return {
        "probability_maps": (
            "preassembled" if probability_source.parent == melodic_dir else "stats"
        ),
        "threshold_maps": "preassembled" if threshold_source.parent == melodic_dir else "stats",
    }


def prepare_task(
    *,
    output_dir: Path,
    bold_file: Path,
    functional_mask: Path,
    confounds_file: Path,
    runner: FSLRunner,
    upstream_dir: Path | None = None,
    anatomical_file: Path | None = None,
    anatomical_mask: Path | None = None,
    gm_probability: Path | None = None,
    wm_probability: Path | None = None,
    csf_probability: Path | None = None,
    melodic_dir: Path | None = None,
    smoothing_retention_mode: str = "revised",
) -> Path:
    """Create CICADA masks, MELODIC products, and component spatial summaries."""
    output_dir = Path(output_dir).resolve()
    inputs = [bold_file, functional_mask, confounds_file]
    optional_inputs = [
        anatomical_file,
        anatomical_mask,
        gm_probability,
        wm_probability,
        csf_probability,
    ]
    for path in [*inputs, *(path for path in optional_inputs if path is not None)]:
        if not Path(path).resolve().is_file():
            raise FileNotFoundError(Path(path).resolve())
    if smoothing_retention_mode not in SMOOTHING_RETENTION_MODES:
        raise ValueError(
            "smoothing_retention_mode must be one of " + ", ".join(SMOOTHING_RETENTION_MODES)
        )
    resolved_melodic = Path(melodic_dir).resolve() if melodic_dir is not None else None
    component_map_sources: dict[str, str] | None = None
    if resolved_melodic is not None:
        component_map_sources = _validate_reusable_melodic_dir(
            resolved_melodic, Path(functional_mask).resolve()
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream = Path(upstream_dir).resolve() if upstream_dir else bundled_upstream_dir()
    script = upstream / "basescripts" / "CICADA_1_MasksandICAs.sh"
    if not script.is_file():
        raise FileNotFoundError(script)

    def value(path: Path | None) -> str:
        return str(Path(path).resolve()) if path is not None else "x"

    command = [
        "env",
        f"CICADA_SMOOTHING_RETENTION_MODE={smoothing_retention_mode}",
        "bash",
        str(script),
        "-o",
        str(output_dir),
        "-F",
        str(Path(bold_file).resolve()),
        "-f",
        str(Path(functional_mask).resolve()),
        "-C",
        str(Path(confounds_file).resolve()),
        "-A",
        value(anatomical_file),
        "-a",
        value(anatomical_mask),
        "-g",
        value(gm_probability),
        "-w",
        value(wm_probability),
        "-c",
        value(csf_probability),
        "-m",
        value(resolved_melodic),
    ]
    runner.run(command, cwd=output_dir)
    actual_melodic = resolved_melodic if resolved_melodic is not None else output_dir / "melodic"
    required = [
        output_dir / "funcfile.nii.gz",
        output_dir / "funcmask.nii.gz",
        output_dir / "ROIcalcs" / "GM_ICmean.txt",
        output_dir / "ROIcalcs" / "IC_exp_variance.txt",
        actual_melodic / "melodic_mix",
        actual_melodic / "ICprobabilities.nii.gz",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("CICADA FSL preparation did not produce: " + ", ".join(missing))

    provenance_dir = output_dir / "cicada_python"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    preparation = {
        "created_utc": datetime.now(UTC).isoformat(),
        "upstream_cicada_commit": UPSTREAM_COMMIT,
        "preparation_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "smoothing_retention_mode": smoothing_retention_mode,
        "melodic_mode": "reused" if resolved_melodic is not None else "generated",
        "component_map_sources": component_map_sources,
        "melodic_dir": str(actual_melodic),
        "fsl_image": str(runner.image) if runner.image is not None else None,
        "fsl_runtime": runner.runtime if runner.image is not None else "host",
        "inputs": {
            "bold_file": str(Path(bold_file).resolve()),
            "functional_mask": str(Path(functional_mask).resolve()),
            "confounds_file": str(Path(confounds_file).resolve()),
        },
        "local_preparation_adaptations": [
            "Reused MELODIC directories are validated and never recursively removed.",
            "Preassembled IC probability and threshold maps are accepted.",
            "Multi-volume component threshold maps retain their final volume.",
            "The network atlas is resampled independently inside every run directory.",
        ],
    }
    (provenance_dir / "preparation.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return actual_melodic
