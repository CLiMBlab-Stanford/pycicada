from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from cicada_python.features import SPATIAL_LABELS, build_features


def _write_prepared_fixture(root: Path, n_components: int = 7, n_volumes: int = 64) -> None:
    roi = root / "ROIcalcs"
    report = root / "melodic" / "report"
    roi.mkdir(parents=True)
    report.mkdir(parents=True)
    affine = np.eye(4)
    bold = np.zeros((3, 3, 3, n_volumes), dtype=np.float32)
    image = nib.Nifti1Image(bold, affine)
    image.header.set_zooms((2.0, 2.0, 2.0, 2.0))
    nib.save(image, root / "funcfile.nii.gz")

    component_scale = np.linspace(0.7, 1.3, n_components)
    for index, label in enumerate(SPATIAL_LABELS, start=1):
        mean = component_scale * (1 + index / 20)
        count = np.full(n_components, 100 + index, dtype=float)
        np.savetxt(roi / f"{label}_ICmean.txt", mean)
        np.savetxt(roi / f"{label}_ICnumvoxels.txt", count)
    np.savetxt(roi / "IC_exp_variance.txt", np.full(n_components, 100 / n_components))

    time = np.arange(n_volumes) * 2.0
    mixing = []
    for component in range(1, n_components + 1):
        series = np.sin(2 * np.pi * (0.01 + component * 0.008) * time) + 0.05 * component
        mixing.append(series)
        np.savetxt(report / f"t{component}.txt", series)
    np.savetxt(root / "melodic" / "melodic_mix", np.column_stack(mixing))

    confounds = pd.DataFrame(
        {
            "framewise_displacement": np.r_[np.nan, np.abs(np.sin(time[1:] / 9))],
            "dvars": np.r_[np.nan, np.abs(np.cos(time[1:] / 11))],
        }
    )
    confounds.to_csv(root / "confounds_timeseries.tsv", sep="\t", index=False, na_rep="n/a")


def test_build_features_from_prepared_products(tmp_path: Path) -> None:
    _write_prepared_fixture(tmp_path)
    result = build_features(tmp_path)
    assert len(result.relative) == 7
    assert result.time_series.shape == (64, 7)
    assert result.repetition_time == 2.0
    assert {"GM", "Smoothing_Retention", "best_power_overlap_norm", "FD_Corr"}.issubset(
        result.relative.columns
    )
    assert np.isfinite(result.relative.drop(columns="ICs").to_numpy()).all()


def test_repetition_time_override_supports_derivatives_with_invalid_header_tr(
    tmp_path: Path,
) -> None:
    _write_prepared_fixture(tmp_path)
    bold_file = tmp_path / "funcfile.nii.gz"
    image = nib.load(bold_file)
    image.header["pixdim"][4] = 0.0
    nib.save(image, bold_file)
    result = build_features(tmp_path, repetition_time=1.08)
    assert result.repetition_time == 1.08


def test_task_events_add_condition_and_best_task_spectral_features(tmp_path: Path) -> None:
    _write_prepared_fixture(tmp_path)
    events = pd.DataFrame(
        {
            "onset": [0.0, 24.0, 48.0],
            "duration": [4.0, 4.0, 4.0],
            "trial_type": ["faces", "places", "faces"],
        }
    )
    events_file = tmp_path / "events.tsv"
    events.to_csv(events_file, sep="\t", index=False)
    result = build_features(tmp_path, events_file=events_file)
    assert {"faces", "places", "Combined", "besttask_power_overlap"}.issubset(
        result.relative.columns
    )
