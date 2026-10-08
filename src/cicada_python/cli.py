"""Command-line interface for pycicada."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .classifier import classify_components
from .features import build_features
from .fsl import FSLRunner
from .pipeline import denoise, write_classification_outputs
from .prepare import SMOOTHING_RETENTION_MODES, bundled_upstream_dir, prepare_task


def _add_fsl_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fsl-image", type=Path)
    parser.add_argument("--runtime", default="singularity")
    parser.add_argument(
        "--bind", action="append", default=[], help="container bind specification; repeatable"
    )
    parser.add_argument(
        "--fsl-setup",
        default=(
            "set +e; source /opt/qunex/env/qunex_environment.sh >/dev/null 2>&1; "
            'qx_setup_rc=$?; set -e; if [ "$qx_setup_rc" -ne 0 ]; then '
            'echo "QuNex environment setup failed with status $qx_setup_rc" >&2; '
            'exit "$qx_setup_rc"; fi'
        ),
        help="shell setup executed inside the FSL container",
    )


def _runner(args: argparse.Namespace) -> FSLRunner:
    return FSLRunner(
        image=args.fsl_image.resolve() if args.fsl_image else None,
        runtime=args.runtime,
        binds=tuple(args.bind),
        inner_setup=args.fsl_setup,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pycicada")
    subcommands = parser.add_subparsers(dest="command", required=True)
    classify = subcommands.add_parser(
        "classify", help="classify and optionally denoise a CICADA-prepared task directory"
    )
    classify.add_argument("--task-dir", type=Path, required=True)
    classify.add_argument("--melodic-dir", type=Path)
    classify.add_argument("--events", type=Path)
    classify.add_argument("--repetition-time", type=float)
    classify.add_argument("--tolerance", type=int, default=5)
    classify.add_argument("--denoise", action="store_true")
    classify.add_argument("--output", type=Path, help="denoised NIfTI output path")
    _add_fsl_options(classify)

    run = subcommands.add_parser("run", help="prepare, classify, and denoise one BOLD run")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--bold", type=Path, required=True)
    run.add_argument("--mask", type=Path, required=True)
    run.add_argument("--confounds", type=Path, required=True)
    run.add_argument("--anat", type=Path)
    run.add_argument("--anat-mask", type=Path)
    run.add_argument("--gm-prob", type=Path)
    run.add_argument("--wm-prob", type=Path)
    run.add_argument("--csf-prob", type=Path)
    run.add_argument("--melodic-dir", type=Path, help="reuse an existing MELODIC directory")
    run.add_argument(
        "--smoothing-retention-mode",
        choices=SMOOTHING_RETENTION_MODES,
        default="revised",
        help="smoothing convention for the CICADA retention feature",
    )
    run.add_argument("--events", type=Path)
    run.add_argument("--repetition-time", type=float)
    run.add_argument("--tolerance", type=int, default=5)
    run.add_argument("--upstream-dir", type=Path)
    run.add_argument("--no-denoise", action="store_true")
    run.add_argument("--output", type=Path, help="denoised NIfTI output path")
    _add_fsl_options(run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runner = _runner(args)
    if args.command == "run":
        task_dir = args.output_dir.resolve()
        task_dir.mkdir(parents=True, exist_ok=True)
        upstream_dir = args.upstream_dir.resolve() if args.upstream_dir else bundled_upstream_dir()
        runner = runner.with_bound_paths(
            [
                task_dir,
                args.bold,
                args.mask,
                args.confounds,
                upstream_dir,
                *(
                    value
                    for value in (
                        args.anat,
                        args.anat_mask,
                        args.gm_prob,
                        args.wm_prob,
                        args.csf_prob,
                        args.melodic_dir,
                    )
                    if value is not None
                ),
            ]
        )
        melodic_dir = prepare_task(
            output_dir=task_dir,
            bold_file=args.bold,
            functional_mask=args.mask,
            confounds_file=args.confounds,
            runner=runner,
            upstream_dir=upstream_dir,
            anatomical_file=args.anat,
            anatomical_mask=args.anat_mask,
            gm_probability=args.gm_prob,
            wm_probability=args.wm_prob,
            csf_probability=args.csf_prob,
            melodic_dir=args.melodic_dir,
            smoothing_retention_mode=args.smoothing_retention_mode,
        )
        should_denoise = not args.no_denoise
    elif args.command == "classify":
        task_dir = args.task_dir.resolve()
        melodic_dir = args.melodic_dir.resolve() if args.melodic_dir else task_dir / "melodic"
        runner = runner.with_bound_paths([task_dir, melodic_dir])
        should_denoise = args.denoise
    else:
        raise AssertionError(args.command)
    features = build_features(
        task_dir,
        melodic_dir,
        events_file=args.events,
        repetition_time=args.repetition_time,
    )
    result = classify_components(features, tolerance=args.tolerance)
    output_dir = write_classification_outputs(
        task_dir, melodic_dir, features, result, tolerance=args.tolerance
    )
    print(f"Classification outputs: {output_dir}")
    print("Signal components: " + ",".join(str(value) for value in result.signal_components))
    print("Noise components: " + ",".join(str(value) for value in result.noise_components))
    for warning in result.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if should_denoise:
        output = denoise(task_dir, melodic_dir, result, runner, output_file=args.output)
        print(f"Denoised BOLD: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
