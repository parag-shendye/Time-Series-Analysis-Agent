"""Unsupervised detectors. Each takes a sensor DataFrame (datetime index) and returns a boolean
Series, True where the sample is anomalous. Extra keyword arguments a detector does not declare are
ignored by `detect`, so callers can pass one shared parameter dict to every detector."""
import inspect
from typing import Callable

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .stats import noise_scale


def mask_to_intervals(mask: pd.Series, min_len: int = 3) -> list[tuple]:
    """Collapse a boolean mask into (first_label, last_label) runs of at least min_len samples."""
    m = mask.astype(bool)
    grp = (m != m.shift()).cumsum()
    return [(g.index[0], g.index[-1]) for _, g in m[m].groupby(grp[m]) if len(g) >= min_len]


def isolation_forest(X: pd.DataFrame, contamination: float = 0.1) -> pd.Series:
    """Flag isolated outlier points. Weak on SKAB, whose faults are sustained level shifts."""
    Z = StandardScaler().fit_transform(X.ffill().bfill())
    pred = IsolationForest(n_estimators=100, contamination=contamination, random_state=0).fit_predict(Z)
    return pd.Series(pred == -1, index=X.index)


def level_shift(X: pd.DataFrame, rel: float = 0.6, z_thr: float = 3.0,
                win: int = 15, min_size: int = 30, penmul: float = 6.0) -> pd.Series:
    """Multivariate level-shift detector.

    Sensors are scaled by a noise estimate taken from first differences, so a level shift does not
    inflate its own scale. After median smoothing, PELT segments the series. A segment is anomalous
    when some key sensor's median differs from the first segment (assumed normal) by more than z_thr
    noise units and by at least `rel` times the largest such shift in the file. Key sensors are the ones
    whose largest shift is at least `rel` times the overall largest, so slow drifts (Temperature,
    Thermocouple), which stay shifted after a fault ends, do not vote.
    """
    S = X.ffill().bfill()
    scale = noise_scale(X).reindex(X.columns).fillna(1.0)  # constant/degenerate sensors: no scaling
    Z = ((S.rolling(win, center=True, min_periods=1).median() - S.median()) / scale).clip(-50, 50).fillna(0.0)
    A = Z.values / Z.values.std(axis=0).clip(min=1e-9)
    if len(A) < 2 * min_size:
        return pd.Series(False, index=X.index)
    bk = rpt.Pelt(model="l2", min_size=min_size, jump=5).fit(A).predict(pen=penmul * np.log(len(A)) * A.shape[1])
    segs = list(zip([0] + bk[:-1], bk))
    base = np.median(Z.values[segs[0][0]:segs[0][1]], axis=0)
    M = np.array([np.abs(np.median(Z.values[a:b], axis=0) - base) for a, b in segs])  # segments x sensors
    key = M.max(axis=0) >= rel * M.max()
    shifts = M[:, key].max(axis=1)
    mask = np.zeros(len(A), bool)
    for (a, b), s in zip(segs, shifts):
        if s > z_thr and s >= rel * shifts.max():
            mask[a:b] = True
    return pd.Series(mask, index=X.index)


def rolling_mean(X: pd.DataFrame, win: int = 30, z_thr: float = 10.0, revert_frac: float = 0.5,
                 min_len: int = 30, max_len: int = 900, min_sensors: int = 1) -> pd.Series:
    """Rolling-mean change detector.

    Per sensor, the change at t is mean(x[t:t+win]) - mean(x[t-win:t]). It is scaled by its own robust
    spread (1.4826 * MAD over the file), so oscillating or drifting sensors automatically need larger
    changes and no sensor is special. A change with |z| > z_thr starts an anomaly; it ends when the
    rolling mean has come back at least `revert_frac` of the way to its pre-change level, within
    `max_len` samples. A change that never reverts is a level step, not a range anomaly, and is not
    flagged. A sample is anomalous when at least `min_sensors` sensors flag it.

    z_thr is a sensitivity/false-alarm dial. On SKAB (win=30): z=7 gives mean F1 0.75 but flags 11% of the
    fault-free recording; z=10 gives F1 0.62 and flags 1.9%, missing 8 of 34 faults entirely. Requiring
    two sensors (min_sensors=2) collapses F1 to about 0.13, since most faults show in one sensor.
    """
    S = X.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    n = len(S)
    votes = np.zeros(n, int)
    for col in S.columns:
        x = S[col]
        before, after = x.rolling(win).mean().shift(1), x.rolling(win).mean().shift(-(win - 1))
        c = after - before
        spread = 1.4826 * (c - c.median()).abs().median()
        if not np.isfinite(spread) or spread <= 1e-12:
            spread = c.std()
        if not np.isfinite(spread) or spread <= 1e-12:
            continue
        z = ((c - c.median()) / spread).abs().to_numpy()
        hot = np.flatnonzero(np.nan_to_num(z) > z_thr)
        if hot.size == 0:
            continue
        smooth = x.rolling(win, center=True, min_periods=1).mean().to_numpy()
        flagged = np.zeros(n, bool)
        for run in np.split(hot, np.flatnonzero(np.diff(hot) > win) + 1):  # one onset per burst of hot samples
            i = int(run[np.argmax(z[run])])
            l0, jump = before.iloc[i], after.iloc[i] - before.iloc[i]
            if not np.isfinite(jump):
                continue
            back = np.flatnonzero(np.abs(smooth[i + min_len:i + max_len] - l0) <= (1 - revert_frac) * abs(jump))
            if back.size:
                flagged[i:i + min_len + int(back[0])] = True
        votes += flagged
    return pd.Series(votes >= min_sensors, index=X.index)


DETECTORS: dict[str, Callable[..., pd.Series]] = {
    "level_shift": level_shift, "rolling_mean": rolling_mean, "isolation_forest": isolation_forest}


def detect(X: pd.DataFrame, detector: str = "level_shift", **params) -> pd.Series:
    """Run a named detector. Parameters the detector does not declare are ignored."""
    if detector not in DETECTORS:
        raise ValueError(f"unknown detector '{detector}'; choose from {list(DETECTORS)}")
    fn = DETECTORS[detector]
    accepted = inspect.signature(fn).parameters
    return fn(X, **{k: v for k, v in params.items() if k in accepted})
