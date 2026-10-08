"""MATLAB-free automatic CICADA component classification."""

from .classifier import ClassificationResult, classify_components
from .features import FeatureSet, build_features

__all__ = ["ClassificationResult", "FeatureSet", "build_features", "classify_components"]
__version__ = "0.1.1"
