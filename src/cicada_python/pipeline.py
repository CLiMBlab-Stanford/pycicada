"""Output and denoising operations for automatic CICADA classification."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from . import __version__
from .classifier import ClassificationResult
from .features import FeatureSet
from .fsl import FSLRunner

UPSTREAM_COMMIT = "b5a97bc3753b1cf3660c0e8add960f46e91ea95c"


def write_classification_outputs(
    task_dir: Path,
    melodic_dir: Path,
    features: FeatureSet,
    result: ClassificationResult,
    *,
    tolerance: int,
) -> Path:
    """Write transparent tabular outputs and component evidence maps."""
    task_dir = Path(task_dir).resolve()
    melodic_dir = Path(melodic_dir).resolve()
    output_dir = task_dir / "cicada_python"
    output_dir.mkdir(parents=True, exist_ok=True)
    features.relative.to_csv(output_dir / "feature_values.tsv", sep="\t", index=False)
    features.general.to_csv(output_dir / "feature_values_general.tsv", sep="\t", index=False)
    result.normalized_features.to_csv(
        output_dir / "feature_values_zscore.tsv", sep="\t", index=False
    )
    result.classifications.to_csv(output_dir / "feature_clusters.tsv", sep="\t", index=False)
    result.component_labels.to_csv(output_dir / "component_labels.tsv", sep="\t", index=False)
    (output_dir / "signal_components.txt").write_text(
        ",".join(str(value) for value in result.signal_components) + "\n", encoding="utf-8"
    )
    (output_dir / "noise_components.txt").write_text(
        ",".join(str(value) for value in result.noise_components) + "\n", encoding="utf-8"
    )

    probability_file = melodic_dir / "ICprobabilities.nii.gz"
    evidence_written = False
    if probability_file.is_file():
        probability_image = nib.load(str(probability_file))
        probability = np.asarray(probability_image.dataobj, dtype=np.float32)
        if probability.ndim == 3:
            probability = probability[..., None]
        expected = len(features.relative)
        if probability.shape[3] != expected:
            raise ValueError(
                f"IC probability volume count ({probability.shape[3]}) does not match "
                f"component count ({expected})"
            )
        signal = np.max(probability[..., result.signal_components - 1], axis=3)
        noise = np.max(probability[..., result.noise_components - 1], axis=3)
        categorical = (signal > 0.949).astype(np.int8) - (
            (noise > 0.949) & ~(signal > 0.949)
        ).astype(np.int8)
        header = probability_image.header.copy()
        header.set_data_dtype(np.float32)
        nib.save(
            nib.Nifti1Image(signal.astype(np.float32), probability_image.affine, header),
            str(output_dir / "signal_ic_overlap.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(noise.astype(np.float32), probability_image.affine, header),
            str(output_dir / "noise_ic_overlap.nii.gz"),
        )
        categorical_header = probability_image.header.copy()
        categorical_header.set_data_dtype(np.int8)
        nib.save(
            nib.Nifti1Image(categorical, probability_image.affine, categorical_header),
            str(output_dir / "signal_noise_evidence.nii.gz"),
        )
        evidence_written = True

    provenance = {
        "pycicada_version": __version__,
        "upstream_cicada_commit": UPSTREAM_COMMIT,
        "created_utc": datetime.now(UTC).isoformat(),
        "task_dir": str(task_dir),
        "melodic_dir": str(melodic_dir),
        "tolerance": tolerance,
        "signal_components_one_based": result.signal_components.tolist(),
        "noise_components_one_based": result.noise_components.tolist(),
        "warnings": list(result.warnings),
        "evidence_maps_written": evidence_written,
        "policy_deviations": [
            "Degenerate feature clusters are handled deterministically.",
            (
                "The fewer-than-two-signal fail-safe retains the two highest-ranked components; "
                "the upstream implementation's assignment appears to retain ICs 1 and 2 instead."
            ),
            "Non-finite intermediate features fail explicitly instead of propagating into k-means.",
        ],
    }
    preparation_file = output_dir / "preparation.json"
    provenance["preparation"] = (
        json.loads(preparation_file.read_text(encoding="utf-8"))
        if preparation_file.is_file()
        else None
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def denoise(
    task_dir: Path,
    melodic_dir: Path,
    result: ClassificationResult,
    runner: FSLRunner,
    *,
    output_file: Path | None = None,
) -> Path:
    """Apply CICADA's nonaggressive noise-component regression with FSL."""
    task_dir = Path(task_dir).resolve()
    melodic_dir = Path(melodic_dir).resolve()
    output_file = (
        Path(output_file).resolve()
        if output_file is not None
        else task_dir / "cleaned" / "desc-cicadaPythonNonAgg_bold.nii.gz"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    noise = ",".join(str(value) for value in result.noise_components)
    runner.run(
        [
            "fsl_regfilt",
            "-i",
            str(task_dir / "funcfile.nii.gz"),
            "-f",
            noise,
            "-d",
            str(melodic_dir / "melodic_mix"),
            "-m",
            str(task_dir / "funcmask.nii.gz"),
            "-o",
            str(output_file),
        ],
        cwd=task_dir,
    )
    if not output_file.is_file() or output_file.stat().st_size == 0:
        raise RuntimeError(f"fsl_regfilt did not create a nonempty output: {output_file}")
    return output_file
