"""Construct the component features used by automatic CICADA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
from scipy.special import gammaln

from .math import (
    column_correlations,
    detrend,
    matlab_round_positive,
    range_normalize,
)

WHOLE_BRAIN = ("fullvolume_smoothed_nothresh", "fullvolume_nothresh", "All")
REGIONS = ("GMWM", "GM", "WM", "Edge", "Subepe", "CSF", "Suscept", "OutbrainOnly")
NOISE_REGIONS = ("Edge", "Subepe", "CSF", "Suscept", "OutbrainOnly")
IN_OUT = ("Inbrain", "Outbrain")
NETWORKS = (
    "MedialVisual",
    "SensoryMotor",
    "DorsalAttention",
    "VentralAttention",
    "FrontoParietal",
    "DefaultModeNetwork",
    "Subcortical",
)
SPATIAL_LABELS = WHOLE_BRAIN + IN_OUT + NETWORKS + REGIONS


@dataclass(frozen=True)
class FeatureSet:
    """Features and supporting values required by CICADA classification."""

    relative: pd.DataFrame
    general: pd.DataFrame
    explained_variance: np.ndarray
    time_series: np.ndarray
    power_spectra: np.ndarray
    repetition_time: float


def _vector(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, name: str) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan),
        where=denominator != 0,
    )
    if not np.all(np.isfinite(result)):
        bad = int(np.count_nonzero(~np.isfinite(result)))
        raise ValueError(f"Feature {name} contains {bad} non-finite values")
    return result


def _load_spatial_tables(task_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roi_dir = task_dir / "ROIcalcs"
    means: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    expected_length: int | None = None
    for label in SPATIAL_LABELS:
        means[label] = _vector(roi_dir / f"{label}_ICmean.txt")
        counts[label] = _vector(roi_dir / f"{label}_ICnumvoxels.txt")
        if expected_length is None:
            expected_length = means[label].size
        if means[label].size != expected_length or counts[label].size != expected_length:
            raise ValueError(f"Spatial feature length mismatch for {label}")
    mean_table = pd.DataFrame(means)
    count_table = pd.DataFrame(counts)
    return mean_table, count_table, mean_table * count_table


def _spatial_features(sums: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roi_general = pd.DataFrame(index=sums.index)
    networks = pd.DataFrame(index=sums.index)
    for label in REGIONS:
        roi_general[label] = _safe_divide(sums[label], sums["All"], label)
    for label in NETWORKS:
        networks[label] = _safe_divide(sums[label], sums["All"], label)
    roi_general["WMCSF"] = roi_general["WM"] + roi_general["CSF"]

    roi_relative = roi_general.copy()
    gm = roi_general["GM"].to_numpy()
    for label in NOISE_REGIONS:
        value = roi_general[label].to_numpy()
        roi_relative[label] = _safe_divide(
            _safe_divide(value, gm + value, f"{label}/(GM+{label})"),
            gm,
            f"relative {label}",
        )
    wm_csf = roi_relative["WM"].to_numpy() * roi_relative["CSF"].to_numpy()
    roi_relative["WMCSF"] = _safe_divide(
        _safe_divide(wm_csf, gm + wm_csf, "WMCSF/(GM+WMCSF)"), gm, "relative WMCSF"
    )

    inout_general = pd.DataFrame(index=sums.index)
    inout_total = sums["Inbrain"].to_numpy() + sums["Outbrain"].to_numpy()
    for label in IN_OUT:
        inout_general[label] = _safe_divide(sums[label], inout_total, label)
    inout_relative = inout_general.copy()
    inout_relative["Outbrain"] = _safe_divide(
        inout_general["Outbrain"],
        inout_general["Inbrain"] + inout_general["Outbrain"],
        "relative Outbrain",
    )

    general = pd.concat([roi_general, inout_general, networks], axis=1)
    relative = pd.concat([roi_relative, inout_relative, networks], axis=1)
    return general, relative


def _spm_hrf(repetition_time: float) -> np.ndarray:
    # CICADA's sampled SPM-style double-gamma response.
    p = np.array([6.0, 16.0, 1.0, 1.0, 6.0, 0.0, 32.0])
    oversampling = 16
    dt = repetition_time / oversampling
    u = np.arange(0, int(p[6] / dt) + 1, dtype=np.float64) - p[5] / dt

    def gamma_density(shape: float, scale_rate: float) -> np.ndarray:
        result = np.zeros_like(u)
        positive = u > 0
        result[positive] = np.exp(
            (shape - 1) * np.log(u[positive])
            + shape * np.log(scale_rate)
            - scale_rate * u[positive]
            - gammaln(shape)
        )
        return result

    hrf_high_res = gamma_density(p[0] / p[2], dt / p[2]) - (
        gamma_density(p[1] / p[3], dt / p[3]) / p[4]
    )
    sample_indices = np.arange(0, int(p[6] / repetition_time) + 1) * oversampling
    hrf = hrf_high_res[sample_indices]
    total = np.sum(hrf)
    if total == 0 or not np.isfinite(total):
        raise ValueError("Could not construct a finite HRF")
    return hrf / total


def _load_component_time_series(melodic_dir: Path, n_components: int, n_volumes: int) -> np.ndarray:
    reports = [melodic_dir / "report" / f"t{i}.txt" for i in range(1, n_components + 1)]
    if all(path.is_file() for path in reports):
        series = np.column_stack([_vector(path) for path in reports])
    else:
        mix = melodic_dir / "melodic_mix"
        if not mix.is_file():
            missing = next((path for path in reports if not path.is_file()), reports[0])
            raise FileNotFoundError(f"Missing {missing} and fallback {mix}")
        series = np.loadtxt(mix, dtype=np.float64, ndmin=2)
    if series.shape == (n_components, n_volumes):
        series = series.T
    if series.shape != (n_volumes, n_components):
        raise ValueError(
            "MELODIC time-series shape mismatch: "
            f"expected {(n_volumes, n_components)}, got {series.shape}"
        )
    return np.asarray(series, dtype=np.float64)


def _spectral_features(
    time_series: np.ndarray, repetition_time: float, hrf: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    n_volumes = time_series.shape[0]
    duration = repetition_time * n_volumes
    sampling_frequency = 1.0 / repetition_time
    frequency_step = 1.0 / duration
    frequencies = sampling_frequency * np.arange(n_volumes // 2 + 1) / n_volumes

    hrf_normalized = (hrf - np.mean(hrf)) / np.std(hrf, ddof=1)
    hrf_fft = np.fft.fft(hrf_normalized)
    hrf_power = np.abs((hrf_fft**2) / hrf.size)[: hrf.size // 2 + 1]
    hrf_power = hrf_power / trapezoid(hrf_power)
    hrf_frequencies = sampling_frequency * np.arange(hrf.size // 2 + 1) / hrf.size
    interpolated = CubicSpline(hrf_frequencies, hrf_power)(frequencies)
    interpolated = interpolated / trapezoid(interpolated)

    transformed = np.fft.fft(time_series, axis=0)
    power = np.abs((transformed**2) / n_volumes)[: n_volumes // 2 + 1]
    areas = trapezoid(power, axis=0)
    normalized_power = _safe_divide(power, areas[None, :], "normalized component power")

    lower = matlab_round_positive(0.008 / frequency_step) + 1
    upper = matlab_round_positive(0.15 / frequency_step) + 1
    lower = min(max(lower, 1), power.shape[0])
    upper = min(max(upper, lower + 1), power.shape[0])
    low = trapezoid(power[:lower], axis=0)
    bold = trapezoid(power[lower : upper - 1], axis=0)
    high = trapezoid(power[upper:], axis=0)
    total = low + bold + high
    frequency_general = pd.DataFrame(
        {
            "Lowfreq": _safe_divide(low, total, "Lowfreq"),
            "BOLDfreq": _safe_divide(bold, total, "BOLDfreq"),
            "Highfreq": _safe_divide(high, total, "Highfreq"),
        }
    )
    frequency_relative = frequency_general.copy()
    frequency_relative["Lowfreq"] = _safe_divide(
        frequency_general["Lowfreq"],
        frequency_general["BOLDfreq"] + frequency_general["Lowfreq"],
        "relative Lowfreq",
    )
    frequency_relative["Highfreq"] = _safe_divide(
        frequency_general["Highfreq"],
        frequency_general["BOLDfreq"] + frequency_general["Highfreq"],
        "relative Highfreq",
    )
    hrf_overlap = pd.DataFrame(
        {"general_power_overlap": np.sum(interpolated[:, None] * normalized_power, axis=0)}
    )
    hrf_difference = pd.DataFrame(
        {
            "general_power_difference": np.sum(
                (interpolated[:, None] - normalized_power) ** 2, axis=0
            )
        }
    )
    return frequency_general, frequency_relative, hrf_overlap, hrf_difference, normalized_power


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=separator, na_values=["n/a", "NA", "NaN"])


def _motion_features(confounds: pd.DataFrame, time_series: np.ndarray) -> pd.DataFrame:
    required = ("framewise_displacement", "dvars")
    missing = [column for column in required if column not in confounds]
    if missing:
        raise ValueError("Missing required confound columns: " + ", ".join(missing))
    if len(confounds) != time_series.shape[0]:
        raise ValueError(
            f"Confound rows ({len(confounds)}) do not match BOLD volumes ({time_series.shape[0]})"
        )
    change = np.abs(np.diff(detrend(time_series, axis=0), axis=0))
    result: dict[str, np.ndarray] = {}
    for column, label in (("framewise_displacement", "FD_Corr"), ("dvars", "DVARS_Corr")):
        values = pd.to_numeric(confounds[column], errors="coerce").to_numpy(dtype=np.float64)[1:]
        if np.any(~np.isfinite(values)):
            bad_rows = (np.flatnonzero(~np.isfinite(values)) + 2).tolist()
            raise ValueError(
                f"Confound {column} has non-finite values after the first row at rows {bad_rows}"
            )
        result[label] = column_correlations(detrend(values), change) ** 2
    return pd.DataFrame(result)


def _task_features(
    events_file: Path,
    time_series: np.ndarray,
    normalized_power: np.ndarray,
    hrf: np.ndarray,
    repetition_time: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    events = _read_table(events_file)
    required = {"onset", "duration", "trial_type"}
    if not required.issubset(events.columns):
        raise ValueError("Events file must contain onset, duration, and trial_type")
    events = events.loc[events["trial_type"].astype(str) != "baseline"].copy()
    conditions = sorted(events["trial_type"].astype(str).unique())
    n_volumes = time_series.shape[0]
    blocks = np.zeros((n_volumes, len(conditions)), dtype=np.float64)
    for index, condition in enumerate(conditions):
        rows = events.loc[events["trial_type"].astype(str) == condition]
        for row in rows.itertuples(index=False):
            start = int(np.ceil(float(row.onset) / repetition_time))
            length = max(1, int(np.ceil(float(row.duration) / repetition_time)))
            stop = min(n_volumes, start + length)
            if start < 0 or start >= n_volumes:
                raise ValueError(f"Event onset falls outside the BOLD series: {row.onset}")
            blocks[start:stop, index] = 1
    padded_hrf = np.pad(hrf, (0, max(0, n_volumes - hrf.size)))[:n_volumes]
    responses = np.column_stack(
        [np.convolve(padded_hrf, blocks[:, i])[:n_volumes] for i in range(len(conditions))]
    )
    responses = np.column_stack([responses, np.sum(responses, axis=1)])
    response_names = [*conditions, "Combined"]
    standardized = np.column_stack(
        [
            (column - np.mean(column)) / np.std(column, ddof=1)
            if np.std(column, ddof=1) > 0
            else np.zeros_like(column)
            for column in responses.T
        ]
    )
    correlations = np.column_stack(
        [
            column_correlations(standardized[:, i], time_series) ** 2
            for i in range(standardized.shape[1])
        ]
    )

    transformed = np.fft.fft(standardized, axis=0)
    task_power = np.abs((transformed**2) / n_volumes)[: n_volumes // 2 + 1]
    task_power = _safe_divide(task_power, trapezoid(task_power, axis=0)[None, :], "task HRF power")
    best = np.argmax(correlations, axis=1)
    matched = task_power[:, best].T
    overlap = np.sum(matched * normalized_power.T, axis=1)
    difference = np.sum((matched - normalized_power.T) ** 2, axis=1)
    return pd.DataFrame(correlations, columns=response_names), overlap, difference


def build_features(
    task_dir: Path,
    melodic_dir: Path | None = None,
    *,
    events_file: Path | None = None,
    repetition_time: float | None = None,
) -> FeatureSet:
    """Build CICADA's per-component feature table from its prepared products."""
    task_dir = Path(task_dir).resolve()
    melodic_dir = Path(melodic_dir).resolve() if melodic_dir else task_dir / "melodic"
    bold_file = task_dir / "funcfile.nii.gz"
    image = nib.load(str(bold_file))
    if len(image.shape) != 4:
        raise ValueError(f"Expected a four-dimensional BOLD file: {bold_file}")
    n_volumes = int(image.shape[3])
    repetition_time = (
        float(repetition_time)
        if repetition_time is not None
        else float(image.header.get_zooms()[3])
    )
    if not np.isfinite(repetition_time) or repetition_time <= 0:
        raise ValueError(f"Invalid repetition time {repetition_time!r} in {bold_file}")

    _means, _counts, sums = _load_spatial_tables(task_dir)
    n_components = len(sums)
    explained = _vector(task_dir / "ROIcalcs" / "IC_exp_variance.txt") / 100.0
    if explained.size != n_components:
        raise ValueError("Explained-variance vector does not match component count")
    time_series = _load_component_time_series(melodic_dir, n_components, n_volumes)
    hrf = _spm_hrf(repetition_time)

    spatial_general, spatial_relative = _spatial_features(sums)
    smoothness = pd.DataFrame(
        {
            "Smoothing_Retention": _safe_divide(
                sums["fullvolume_smoothed_nothresh"],
                sums["fullvolume_nothresh"],
                "Smoothing_Retention",
            )
        }
    )
    frequency_general, frequency_relative, overlap, difference, powers = _spectral_features(
        time_series, repetition_time, hrf
    )
    confounds_candidates = [
        task_dir / "confounds_timeseries.tsv",
        task_dir / "confounds_timeseries.csv",
    ]
    confounds_path = next((path for path in confounds_candidates if path.is_file()), None)
    if confounds_path is None:
        raise FileNotFoundError("Could not find confounds_timeseries.tsv or .csv")
    motion = _motion_features(_read_table(confounds_path), time_series)
    spikiness = pd.DataFrame({"Spikiness": np.max(np.abs(time_series), axis=0)})

    condition_correlations = pd.DataFrame(index=np.arange(n_components))
    if events_file is not None:
        condition_correlations, task_overlap, task_difference = _task_features(
            Path(events_file), time_series, powers, hrf, repetition_time
        )
        overlap["besttask_power_overlap"] = task_overlap
        difference["besttask_power_difference"] = task_difference

    overlap_scaled = range_normalize(overlap.to_numpy(), axis=0)
    difference_scaled = range_normalize(difference.to_numpy(), axis=0)
    overlap["best_power_overlap_norm"] = np.max(overlap_scaled, axis=1)
    difference["best_power_difference_norm"] = np.min(difference_scaled, axis=1)

    ic = pd.DataFrame({"ICs": np.arange(1, n_components + 1, dtype=int)})
    general = pd.concat(
        [
            ic,
            spatial_general,
            frequency_general,
            motion,
            smoothness,
            overlap,
            difference,
            spikiness,
            condition_correlations,
        ],
        axis=1,
    )
    relative = pd.concat(
        [
            ic,
            spatial_relative,
            frequency_relative,
            motion,
            smoothness,
            overlap,
            difference,
            spikiness,
            condition_correlations,
        ],
        axis=1,
    )
    numeric = relative.drop(columns="ICs").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        columns = relative.drop(columns="ICs").columns[np.any(~np.isfinite(numeric), axis=0)]
        raise ValueError("Non-finite final features: " + ", ".join(columns))
    return FeatureSet(relative, general, explained, time_series, powers, repetition_time)
