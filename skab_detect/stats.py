"""Shared robust statistics."""
import numpy as np
import pandas as pd

MAD_NORMAL = 0.67448975


def noise_scale(X: pd.DataFrame) -> pd.Series:
    """Estimate per-sensor noise sigma from valid consecutive differences.

    Uses the MAD of first differences, assuming approximately Gaussian
    measurement noise. Does not bridge missing observations. Differencing
    cancels level shifts and slow drifts, so a fault does not inflate its own
    scale the way a whole-file std does. Sensors with no usable estimate get NaN.
    """
    result = {}

    for col in X.columns:
        s = pd.to_numeric(X[col], errors="coerce").astype(float)

        # Only use differences between consecutive valid observations.
        valid_pairs = s.notna() & s.shift(1).notna()
        differences = s.diff()[valid_pairs].abs()

        if len(differences) < 10:
            result[col] = np.nan
            continue

        scale = differences.median() / (MAD_NORMAL * np.sqrt(2))

        # Fallback: robust spread of the original signal.
        # This is a fallback, not an equivalent noise estimate.
        if not np.isfinite(scale) or scale <= 1e-12:
            median = s.median()
            signal_mad = (s - median).abs().median()
            scale = signal_mad / MAD_NORMAL

        if not np.isfinite(scale) or scale <= 1e-12:
            result[col] = np.nan
        else:
            result[col] = float(scale)

    return pd.Series(result, dtype=float)
