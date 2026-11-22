"""Compare a candidate window with the rest of the file."""
import numpy as np
import pandas as pd

from .stats import noise_scale

SHIFT_THRESHOLD = 3.0  # noise units
# Slow thermal sensors drift steadily through every file. Measured on 20 SKAB valve1/valve2 faults vs 20
# normal windows of the same length, their median |shift| was no larger in the faults (Thermocouple 5.1 vs
# 8.1 noise units, Temperature 5.8 vs 10.2; Volume Flow 67 vs 0.0), so a shift there says nothing about a fault.
SLOW_SENSORS = ("Temperature", "Thermocouple")


def compare_window(X: pd.DataFrame, start, end, threshold: float = SHIFT_THRESHOLD,
                   slow_sensors: tuple = SLOW_SENSORS) -> dict:
    """Per-sensor shift of the median inside [start, end] versus the median outside it, in noise units.

    Sensors with |shift| above `threshold` are listed largest first; the rest are in `unchanged`.
    `slow_sensors` (Temperature, Thermocouple) drift through the whole file and shift as much in normal
    stretches as in faults, so they are never listed as evidence: if they exceed the threshold their names
    go in `slow_drift_not_evidence`, otherwise in `unchanged`.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start must be <= end")
    try:
        inside = (X.index >= start_ts) & (X.index <= end_ts)
    except TypeError as exc:
        raise ValueError("window timestamps must have the same timezone convention as the dataset index") from exc
    if not inside.any():
        raise ValueError("no samples in that window")
    if inside.all():
        raise ValueError("window covers the whole file; there is nothing to compare it with")

    W = X.loc[inside].apply(pd.to_numeric, errors="coerce")
    rest = X.loc[~inside].apply(pd.to_numeric, errors="coerce")
    shifts = ((W.median() - rest.median()) / noise_scale(X)).replace([np.inf, -np.inf], np.nan).dropna()
    over = shifts[shifts.abs() > threshold]
    slow = [c for c in over.index if c in slow_sensors]
    shifted = over.drop(slow).sort_values(key=lambda x: x.abs(), ascending=False)
    return {
        "start": str(start_ts), "end": str(end_ts),
        "n_window": int(inside.sum()), "n_outside": int((~inside).sum()),
        "shifted": [{"sensor": c, "window_median": round(float(W[c].median()), 3),
                     "outside_median": round(float(rest[c].median()), 3),
                     "shift_noise_units": round(float(shifts[c]), 2),
                     "n_window_valid": int(W[c].notna().sum()),
                     "n_outside_valid": int(rest[c].notna().sum())} for c in shifted.index],
        "slow_drift_not_evidence": slow,
        "unchanged": [c for c in X.columns if c not in shifted.index and c not in slow],
    }
