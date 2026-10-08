import numpy as np
import pandas as pd
import pytest

from cicada_python.classifier import classify_components
from cicada_python.features import NETWORKS, FeatureSet


def _feature_set(n: int = 9) -> FeatureSet:
    ascending = np.linspace(0.1, 0.9, n)
    descending = ascending[::-1]
    data: dict[str, np.ndarray] = {
        "ICs": np.arange(1, n + 1),
        "GMWM": ascending,
        "GM": ascending,
        "WM": descending,
        "Edge": descending,
        "Subepe": descending,
        "CSF": descending,
        "Suscept": descending,
        "OutbrainOnly": descending,
        "WMCSF": descending,
        "Inbrain": ascending,
        "Outbrain": descending,
    }
    for network in NETWORKS:
        data[network] = ascending
    data.update(
        {
            "Lowfreq": descending,
            "BOLDfreq": ascending,
            "Highfreq": descending,
            "FD_Corr": descending,
            "DVARS_Corr": descending,
            "Smoothing_Retention": ascending,
            "general_power_overlap": ascending,
            "best_power_overlap_norm": ascending,
            "general_power_difference": descending,
            "best_power_difference_norm": descending,
            "Spikiness": descending * 4,
        }
    )
    frame = pd.DataFrame(data)
    return FeatureSet(
        relative=frame,
        general=frame.copy(),
        explained_variance=np.full(n, 1 / n),
        time_series=np.zeros((40, n)),
        power_spectra=np.zeros((21, n)),
        repetition_time=2.0,
    )


def test_classifier_partitions_components_and_preserves_one_based_ids() -> None:
    result = classify_components(_feature_set(), tolerance=5)
    combined = np.sort(np.concatenate([result.signal_components, result.noise_components]))
    np.testing.assert_array_equal(combined, np.arange(1, 10))
    assert not set(result.signal_components) & set(result.noise_components)
    assert result.ranking[0] == 9
    assert set(result.component_labels["PotentialICs"]) == set(range(1, 10))


def test_fail_safe_keeps_highest_ranked_not_component_numbers_one_and_two() -> None:
    features = _feature_set()
    # Make every noise marker rise with the otherwise signal-like features.
    for column in ("Edge", "Subepe", "CSF", "Suscept", "OutbrainOnly", "Outbrain"):
        features.relative[column] = np.linspace(0.1, 0.9, len(features.relative))
    result = classify_components(features, tolerance=1)
    if result.warnings:
        np.testing.assert_array_equal(result.signal_components, np.sort(result.ranking[:2]))


def test_classifier_rejects_decomposition_with_no_possible_noise_component() -> None:
    with pytest.raises(ValueError, match="at least three"):
        classify_components(_feature_set(2))
