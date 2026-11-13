"""Candidate changepoints for a single sensor."""
import numpy as np
import pandas as pd
import ruptures as rpt


def find_changepoints(s: pd.Series, n_changepoints: int = 3, min_size: int = 10) -> dict:
    """Binary segmentation (l2) on a robustly standardised series.

    This is fixed-count segmentation: treat the result as candidate boundaries, not confirmed events.
    Returns {"changepoints": [{ts, mean_before, mean_after, delta}], "method", "n_valid"} or, when no
    segmentation is possible, {"changepoints": [], "reason": ...}.
    """
    s = pd.to_numeric(s, errors="coerce").astype(float)
    valid = s.dropna()
    if len(valid) < 30:
        raise ValueError("at least 30 valid samples are required")
    if valid.nunique() <= 1:
        return {"changepoints": [], "reason": "constant signal"}

    values = valid.to_numpy(dtype=float)
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.std(values)
    if not np.isfinite(scale) or scale < 1e-12:
        return {"changepoints": [], "reason": "signal has negligible variation"}

    z = ((values - median) / scale).reshape(-1, 1)
    n = max(1, min(int(n_changepoints), 10))
    max_bkps = min(n, max(0, len(z) // min_size - 1))
    if max_bkps == 0:
        return {"changepoints": [], "reason": "series too short for segmentation"}

    bkps = rpt.Binseg(model="l2", min_size=min_size, jump=1).fit(z).predict(n_bkps=max_bkps)[:-1]
    out = []
    for b in bkps:
        before = valid.iloc[max(0, b - 20):b]
        after = valid.iloc[b:min(len(valid), b + 20)]
        if before.empty or after.empty:
            continue
        out.append({"ts": str(valid.index[b]), "mean_before": round(float(before.mean()), 3),
                    "mean_after": round(float(after.mean()), 3),
                    "delta": round(float(after.mean() - before.mean()), 3)})
    return {"changepoints": out, "method": "binary_segmentation_l2", "n_valid": int(len(valid))}
