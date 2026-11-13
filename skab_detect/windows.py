"""Compare a candidate window with the rest of the file."""
import numpy as np
import pandas as pd

from .stats import noise_scale

SHIFT_THRESHOLD = 3.0  # noise units


def compare_window(X: pd.DataFrame, start, end, threshold: float = SHIFT_THRESHOLD) -> dict:
    """Per-sensor shift of the median inside [start, end] versus the median outside it, in noise units.

    Sensors with |shift| above `threshold` are listed largest first; the rest are in `unchanged`.
    A bigger shift is stronger evidence but not a reliable pass/fail test: Temperature and Thermocouple
    drift slowly and can shift even in normal stretches.
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
    shifted = shifts[shifts.abs() > threshold].sort_values(key=lambda x: x.abs(), ascending=False)
    return {
        "start": str(start_ts), "end": str(end_ts),
        "n_window": int(inside.sum()), "n_outside": int((~inside).sum()),
        "shifted": [{"sensor": c, "window_median": round(float(W[c].median()), 3),
                     "outside_median": round(float(rest[c].median()), 3),
                     "shift": round(float(shifts[c]), 2),
                     "n_window_valid": int(W[c].notna().sum()),
                     "n_outside_valid": int(rest[c].notna().sum())} for c in shifted.index],
        "unchanged": [c for c in X.columns if c not in shifted.index],
    }
