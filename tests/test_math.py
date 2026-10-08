import numpy as np

from cicada_python.math import high_low_clusters, matlab_round_positive, range_normalize


def test_matlab_round_positive_uses_half_up() -> None:
    assert matlab_round_positive(2.5) == 3
    assert matlab_round_positive(2.49) == 2


def test_range_normalize_constant_is_zero() -> None:
    actual = range_normalize(np.array([4.0, 4.0, 4.0]))
    np.testing.assert_array_equal(actual, np.zeros(3))


def test_high_low_clusters_follow_ordered_three_group_design() -> None:
    values = np.array([0.0, 0.1, 0.2, 4.0, 4.2, 9.0, 9.1])
    low, high = high_low_clusters(values)
    assert set(np.flatnonzero(low)) == {0, 1, 2}
    assert set(np.flatnonzero(high)) == {5, 6}


def test_constant_cluster_has_no_artificial_high_or_low_group() -> None:
    low, high = high_low_clusters(np.ones(5))
    assert not low.any()
    assert not high.any()
