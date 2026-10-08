from pathlib import Path

from cicada_python.fsl import FSLRunner


def test_container_command_quotes_arguments_and_sets_working_directory(tmp_path: Path) -> None:
    image = tmp_path / "image.sif"
    image.touch()
    runner = FSLRunner(image=image, inner_setup="true")
    command = runner.command(
        ["fslmaths", "/path with spaces/in.nii.gz", "-Tmean", "out"], cwd=tmp_path
    )
    assert command[:3] == ["singularity", "exec", "--cleanenv"]
    assert ["--pwd", str(tmp_path)] == command[3:5]
    assert "'/path with spaces/in.nii.gz'" in command[-1]


def test_bound_paths_adds_file_parent_and_directory(tmp_path: Path) -> None:
    image = tmp_path / "image.sif"
    image.touch()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "bold.nii.gz"
    data_file.touch()
    runner = FSLRunner(image=image).with_bound_paths([data_file, data_dir])
    expected = f"{data_dir}:{data_dir}"
    assert runner.binds.count(expected) == 1
