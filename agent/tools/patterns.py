from skab_detect import compare_window, find_changepoints

from .registry import ctx, tool


@tool
def detect_changepoints(sensor: str, n_changepoints: int = 3) -> dict:
    """Find candidate abrupt changes in a sensor using binary segmentation.

    Args:
        sensor: Sensor column name, e.g. Current.
        n_changepoints: Maximum number of changepoints to return (1-10).
    """
    X = ctx.series.X
    if sensor not in X.columns:
        raise ValueError(f"unknown sensor '{sensor}'; choose from {list(X.columns)}")
    return {"sensor": sensor, **find_changepoints(X[sensor], n_changepoints)}


@tool
def extract_window_features(start: str, end: str) -> dict:
    """Compare sensor medians inside a candidate window with the rest of the file.

    `shift_noise_units` is the median difference divided by the sensor's noise (not raw units). Temperature
    and Thermocouple drift slowly in every file, so they are reported under slow_drift_not_evidence and
    must not be used as evidence for or against an anomaly.

    Args:
        start: Window start timestamp.
        end: Window end timestamp.
    """
    return compare_window(ctx.series.X, start, end)
