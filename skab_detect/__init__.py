"""skab_detect: anomaly detection for SKAB-style multivariate sensor data.

Pure pandas/numpy/scikit-learn/ruptures. No LLM, database or MCP dependencies, so everything here can be
tested and scored without a model. The agent (and later an MCP server) wrap these functions.
"""
from .changepoints import find_changepoints
from .detectors import DETECTORS, detect, mask_to_intervals
from .metrics import point_scores
from .stats import noise_scale
from .windows import compare_window

__all__ = ["DETECTORS", "compare_window", "detect", "find_changepoints", "mask_to_intervals",
           "noise_scale", "point_scores"]
