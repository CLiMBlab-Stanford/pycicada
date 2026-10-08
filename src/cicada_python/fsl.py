"""Small, explicit execution boundary for FSL commands."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class FSLRunner:
    """Run FSL either on the host or in a Singularity/Apptainer image."""

    image: Path | None = None
    runtime: str = "singularity"
    binds: tuple[str, ...] = ()
    inner_setup: str = (
        "set +e; source /opt/qunex/env/qunex_environment.sh >/dev/null 2>&1; "
        'qx_setup_rc=$?; set -e; if [ "$qx_setup_rc" -ne 0 ]; then '
        'echo "QuNex environment setup failed with status $qx_setup_rc" >&2; '
        'exit "$qx_setup_rc"; fi'
    )

    def command(self, args: Sequence[str], *, cwd: Path | None = None) -> list[str]:
        values = [str(value) for value in args]
        if self.image is None:
            executable = shutil.which(values[0])
            if executable is None:
                raise RuntimeError(f"FSL command is not available on PATH: {values[0]}")
            values[0] = executable
            return values
        prefix = [self.runtime, "exec", "--cleanenv"]
        for bind in self.binds:
            prefix.extend(["--bind", bind])
        if cwd is not None:
            prefix.extend(["--pwd", str(Path(cwd).resolve())])
        script = "set -e; " + self.inner_setup + "; exec " + shlex.join(values)
        return [*prefix, str(self.image), "bash", "-lc", script]

    def with_bound_paths(self, paths: Sequence[Path]) -> FSLRunner:
        """Bind existing host paths at identical locations when containerized."""
        if self.image is None:
            return self
        binds = list(self.binds)
        sources = {item.split(":", 1)[0] for item in binds}
        for raw_path in paths:
            path = Path(raw_path).resolve()
            if path.is_file():
                path = path.parent
            if not path.exists():
                path = path.parent
            value = str(path)
            if value not in sources:
                binds.append(f"{value}:{value}")
                sources.add(value)
        return replace(self, binds=tuple(binds))

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = self.command(args, cwd=cwd)
        environment = dict(os.environ)
        environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"
        return subprocess.run(
            command,
            cwd=None if self.image is not None else cwd,
            env=environment,
            check=True,
            text=True,
            capture_output=capture_output,
        )

    def describe(self, args: Sequence[str]) -> str:
        return shlex.join(self.command(args))
