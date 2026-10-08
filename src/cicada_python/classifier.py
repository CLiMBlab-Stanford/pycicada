"""CICADA's rule-based automatic component classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import NETWORKS, FeatureSet
from .math import high_low_clusters, range_normalize, zscore


@dataclass(frozen=True)
class ClassificationResult:
    """Automatic CICADA labels and their interpretable intermediate tables."""

    component_labels: pd.DataFrame
    classifications: pd.DataFrame
    normalized_features: pd.DataFrame
    signal_components: np.ndarray
    noise_components: np.ndarray
    ranking: np.ndarray
    warnings: tuple[str, ...]


def _tags(row: pd.Series, columns: list[str]) -> str:
    selected = [column for column in columns if bool(row[column])]
    return "; ".join(selected) if selected else "None"


def classify_components(features: FeatureSet, *, tolerance: int = 5) -> ClassificationResult:
    """Apply CICADA's high/middle/low tags, ranking, and stopping policy."""
    if tolerance < 1:
        raise ValueError("Tolerance must be at least 1")
    relative = features.relative.reset_index(drop=True)
    n_components = len(relative)
    if n_components < 3:
        raise ValueError("CICADA classification requires at least three components")

    classification = pd.DataFrame({"ICs": relative["ICs"].astype(int)})
    for column in relative.columns:
        if column == "ICs":
            continue
        low, high = high_low_clusters(relative[column].to_numpy(dtype=np.float64))
        classification[f"High_{column}"] = high
        classification[f"Low_{column}"] = low

    condition_names = [
        column
        for column in relative.columns
        if column
        not in {
            "ICs",
            "GMWM",
            "GM",
            "WM",
            "Edge",
            "Subepe",
            "CSF",
            "Suscept",
            "OutbrainOnly",
            "WMCSF",
            "Inbrain",
            "Outbrain",
            *NETWORKS,
            "Lowfreq",
            "BOLDfreq",
            "Highfreq",
            "FD_Corr",
            "DVARS_Corr",
            "Smoothing_Retention",
            "general_power_overlap",
            "besttask_power_overlap",
            "best_power_overlap_norm",
            "general_power_difference",
            "besttask_power_difference",
            "best_power_difference_norm",
            "Spikiness",
        }
    ]
    has_task_features = "besttask_power_overlap" in relative.columns
    if not has_task_features:
        classification["High_besttask_power_overlap"] = classification["High_general_power_overlap"]

    classification["High_Subepe"] = classification["High_Subepe"] | classification["High_WMCSF"]
    classification["High_Spikiness"] = classification["High_Spikiness"] & (
        relative["Spikiness"] > 5
    )

    best = ["High_GM", "High_Smoothing_Retention", "High_best_power_overlap_norm"]
    other_good = ["High_general_power_overlap"]
    if has_task_features:
        other_good.append("High_besttask_power_overlap")
    good = [*best, *other_good]
    worst = ["Low_GM", "Low_best_power_overlap_norm"]
    bad_region = [
        "Low_GM",
        "High_Edge",
        "High_Subepe",
        "High_CSF",
        "High_Suscept",
        "High_OutbrainOnly",
    ]
    bad = [
        *bad_region,
        "High_Outbrain",
        "High_Highfreq",
        "High_Spikiness",
        "Low_best_power_overlap_norm",
        "Low_Smoothing_Retention",
        "High_DVARS_Corr",
        "High_FD_Corr",
    ]
    bad = list(dict.fromkeys(bad))
    network_tags = [f"High_{name}" for name in NETWORKS]
    condition_tags = [f"High_{name}" for name in condition_names]

    score = (
        range_normalize(relative["Smoothing_Retention"].to_numpy())
        * range_normalize(relative["GM"].to_numpy()) ** 2
        * range_normalize(relative["best_power_overlap_norm"].to_numpy())
    )
    # Stable sorting makes tied decisions reproducible and favors lower IC numbers.
    ranking = np.argsort(-score, kind="stable")
    candidates_above_mean = int(np.count_nonzero(score > np.mean(score)))
    stop_budget = tolerance
    signal = np.zeros(n_components, dtype=bool)
    considered: list[int] = []

    for index in ranking[:candidates_above_mean]:
        if stop_budget <= 0:
            break
        considered.append(int(index))
        has_primary_evidence = bool(
            classification.loc[index, "High_GM"]
            or classification.loc[index, "High_best_power_overlap_norm"]
            or classification.loc[index, "High_besttask_power_overlap"]
        )
        lacks_worst_tags = not bool(classification.loc[index, worst].any())
        region_is_acceptable = bool(
            classification.loc[index, "High_GM"] or not classification.loc[index, bad_region].any()
        )
        strong_gm_case = bool(
            classification.loc[index, "High_GM"]
            and (
                classification.loc[index, "High_best_power_overlap_norm"]
                or classification.loc[index, "High_Smoothing_Retention"]
            )
        )
        only_bad_region_tags = int(classification.loc[index, bad].sum()) == int(
            classification.loc[index, bad_region].sum()
        )
        keep = (
            has_primary_evidence
            and lacks_worst_tags
            and region_is_acceptable
            and (strong_gm_case or only_bad_region_tags)
        )
        signal[index] = keep
        stop_budget = min(tolerance, stop_budget + 1) if keep else stop_budget - 1

    warnings: list[str] = []
    if np.count_nonzero(signal) < 2:
        warnings.append(
            "Fewer than two components met CICADA's signal rules; the two highest-ranked "
            "components were retained as a fail-safe. Treat this run as a QC failure."
        )
        signal[:] = False
        signal[ranking[:2]] = True

    signal_components = np.flatnonzero(signal) + 1
    noise_components = np.flatnonzero(~signal) + 1
    high_signal = classification[best].all(axis=1).to_numpy()
    high_noise = classification[bad].any(axis=1).to_numpy()

    ordered = ranking
    component_labels = pd.DataFrame(
        {
            "PotentialICs": ordered + 1,
            "SignalLabel": signal[ordered].astype(int),
            "HighSignalLabel": high_signal[ordered].astype(int),
            "HighNoiseLabel": high_noise[ordered].astype(int),
            "ConsideredBeforeStop": np.isin(ordered, considered).astype(int),
            "RankingScore": score[ordered],
        }
    )
    component_labels["Good_Tags"] = [_tags(classification.loc[index], good) for index in ordered]
    component_labels["Bad_Tags"] = [_tags(classification.loc[index], bad) for index in ordered]
    component_labels["Network_Tags"] = [
        _tags(classification.loc[index], network_tags) for index in ordered
    ]
    if condition_tags:
        component_labels["Condition_Tags"] = [
            _tags(classification.loc[index], condition_tags) for index in ordered
        ]

    normalized_values = zscore(relative.drop(columns="ICs").to_numpy(), axis=0)
    normalized = pd.DataFrame(normalized_values, columns=relative.columns[1:])
    normalized.insert(0, "ICs", relative["ICs"].to_numpy())
    return ClassificationResult(
        component_labels=component_labels,
        classifications=classification,
        normalized_features=normalized,
        signal_components=signal_components,
        noise_components=noise_components,
        ranking=ranking + 1,
        warnings=tuple(warnings),
    )
