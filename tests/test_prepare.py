import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from cicada_python.prepare import _validate_reusable_melodic_dir, prepare_task


def _image(path: Path, affine: np.ndarray | None = None) -> None:
    nib.save(
        nib.Nifti1Image(
            np.zeros((3, 3, 3, 4), dtype=np.float32),
            np.eye(4) if affine is None else affine,
        ),
        path,
    )


def _valid_melodic(root: Path) -> Path:
    melodic = root / "melodic"
    melodic.mkdir()
    _image(melodic / "melodic_IC.nii.gz")
    _image(melodic / "ICprobabilities.nii.gz")
    _image(melodic / "ICthresh_zstat.nii.gz")
    (melodic / "melodic_mix").write_text("0 0 0 0\n", encoding="utf-8")
    (melodic / "melodic_ICstats").write_text("25\n", encoding="utf-8")
    return melodic


def test_reused_melodic_validation_accepts_complete_matching_directory(tmp_path: Path) -> None:
    mask = tmp_path / "mask.nii.gz"
    _image(mask)
    melodic = _valid_melodic(tmp_path)
    sources = _validate_reusable_melodic_dir(melodic, mask)
    assert sources == {"probability_maps": "preassembled", "threshold_maps": "preassembled"}


def test_reused_melodic_validation_does_not_remove_incomplete_directory(tmp_path: Path) -> None:
    mask = tmp_path / "mask.nii.gz"
    _image(mask)
    melodic = _valid_melodic(tmp_path)
    sentinel = melodic / "keep-me"
    sentinel.touch()
    (melodic / "melodic_IC.nii.gz").unlink()
    with pytest.raises(ValueError, match="Refusing to use incomplete"):
        _validate_reusable_melodic_dir(melodic, mask)
    assert sentinel.exists()


def test_reused_melodic_validation_rejects_geometry_mismatch(tmp_path: Path) -> None:
    mask = tmp_path / "mask.nii.gz"
    _image(mask)
    melodic = _valid_melodic(tmp_path)
    shifted = np.eye(4)
    shifted[0, 3] = 10
    _image(melodic / "ICprobabilities.nii.gz", shifted)
    with pytest.raises(ValueError, match="geometry does not match"):
        _validate_reusable_melodic_dir(melodic, mask)


def test_prepare_records_explicit_smoothing_and_reuse_provenance(tmp_path: Path) -> None:
    mask = tmp_path / "mask.nii.gz"
    bold = tmp_path / "bold.nii.gz"
    confounds = tmp_path / "confounds.tsv"
    _image(mask)
    _image(bold)
    confounds.write_text("framewise_displacement\tdvars\n0\t0\n", encoding="utf-8")
    melodic = _valid_melodic(tmp_path)
    output = tmp_path / "output"

    class FakeRunner:
        image = None
        runtime = "host"
        command: list[str] | None = None

        def run(self, args: list[str], *, cwd: Path) -> None:
            self.command = args
            (cwd / "ROIcalcs").mkdir(parents=True)
            for path in (
                cwd / "funcfile.nii.gz",
                cwd / "funcmask.nii.gz",
                cwd / "ROIcalcs/GM_ICmean.txt",
                cwd / "ROIcalcs/IC_exp_variance.txt",
            ):
                path.write_text("1\n", encoding="utf-8")

    runner = FakeRunner()
    prepare_task(
        output_dir=output,
        bold_file=bold,
        functional_mask=mask,
        confounds_file=confounds,
        runner=runner,  # type: ignore[arg-type]
        melodic_dir=melodic,
        smoothing_retention_mode="historical",
    )
    assert runner.command is not None
    assert runner.command[:3] == [
        "env",
        "CICADA_SMOOTHING_RETENTION_MODE=historical",
        "bash",
    ]
    provenance = json.loads((output / "cicada_python/preparation.json").read_text())
    assert provenance["smoothing_retention_mode"] == "historical"
    assert provenance["melodic_mode"] == "reused"
    assert provenance["component_map_sources"]["probability_maps"] == "preassembled"
