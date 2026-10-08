"""Numerical helpers expressing the mathematical conventions used by CICADA."""

from __future__ import annotations

import numpy as np
from scipy import signal


def zscore(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standardize using the sample standard deviation, as MATLAB normalize does."""
    array = np.asarray(values, dtype=np.float64)
    mean = np.mean(array, axis=axis, keepdims=True)
    scale = np.std(array, axis=axis, ddof=1, keepdims=True)
    return np.divide(array - mean, scale, out=np.zeros_like(array), where=scale > 0)


def range_normalize(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Map each finite feature to [0, 1], leaving constant features at zero."""
    array = np.asarray(values, dtype=np.float64)
    low = np.min(array, axis=axis, keepdims=True)
    span = np.max(array, axis=axis, keepdims=True) - low
    return np.divide(array - low, span, out=np.zeros_like(array), where=span > 0)


def matlab_round_positive(value: float) -> int:
    """Round a nonnegative value to nearest integer with half values rounded up."""
    if value < 0 or not np.isfinite(value):
        raise ValueError(f"Expected a finite nonnegative value, got {value!r}")
    return int(np.floor(value + 0.5))


def detrend(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Remove a least-squares linear trend along an axis."""
    return signal.detrend(np.asarray(values, dtype=np.float64), axis=axis, type="linear")


def column_correlations(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation between one vector and every matrix column."""
    x = np.asarray(vector, dtype=np.float64).reshape(-1)
    y = np.asarray(matrix, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] != x.size:
        raise ValueError(f"Correlation length mismatch: {x.size} != {y.shape[0]}")
    x = x - np.mean(x)
    y = y - np.mean(y, axis=0, keepdims=True)
    numerator = x @ y
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y, axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(y.shape[1], dtype=np.float64),
        where=denominator > 0,
    )


def high_low_clusters(values: np.ndarray, *, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return low/high memberships from CICADA's one-dimensional 3-group design.

    CICADA initializes k-means at the minimum, median, and maximum. This direct
    implementation preserves that scientific choice while making ties and
    degenerate features deterministic. Empty clusters retain their previous
    centroid; a constant feature is neither high nor low.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Clustering features must be nonempty and finite")
    if np.ptp(x) == 0:
        empty = np.zeros(x.size, dtype=bool)
        return empty.copy(), empty

    centroids = np.array([np.min(x), np.median(x), np.max(x)], dtype=np.float64)
    labels = np.zeros(x.size, dtype=np.int64)
    for _ in range(max_iter):
        new_labels = np.argmin(np.abs(x[:, None] - centroids[None, :]), axis=1)
        updated = centroids.copy()
        for cluster in range(3):
            members = x[new_labels == cluster]
            if members.size:
                updated[cluster] = np.mean(members)
        if np.array_equal(new_labels, labels) and np.allclose(updated, centroids):
            labels = new_labels
            centroids = updated
            break
        labels = new_labels
        centroids = updated

    low_cluster = int(np.argmin(centroids))
    high_cluster = int(np.argmax(centroids))
    return labels == low_cluster, labels == high_cluster
