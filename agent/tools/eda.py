import numpy as np
import pandas as pd

from .registry import ctx, tool


def _sensor(sensor: str) -> pd.Series:
    X = ctx.series.X

    if sensor not in X.columns:
        raise ValueError(
            f"Unknown sensor '{sensor}'. Available sensors: {list(X.columns)}"
        )

    return X[sensor].astype(float)


def _sig(x: float, digits: int = 4) -> float:
    """Round to significant digits (sensors range from 0.03 to 250, so fixed decimals don't fit)."""
    return float(f"{float(x):.{digits}g}")


def _timestamp_info(index: pd.Index) -> dict:
    result = {
        "index_type": type(index).__name__,
        "duplicate_timestamps": int(index.duplicated().sum()),
        "monotonic": bool(index.is_monotonic_increasing),
    }

    if isinstance(index, pd.DatetimeIndex):
        result["datetime_index"] = True

        if len(index) > 1:
            diffs = index.to_series().diff().dropna()
            result["sampling_interval"] = str(diffs.median())
            result["sampling_regular"] = bool(
                diffs.nunique() == 1
            )
        else:
            result["sampling_interval"] = None
            result["sampling_regular"] = None
    else:
        result["datetime_index"] = False
        result["sampling_interval"] = None
        result["sampling_regular"] = None

    return result


@tool
def describe_series(sensor: str = "") -> dict:
    """Describe one sensor or all sensors in the loaded dataset.

    Returns descriptive statistics, missing values, trend,
    constant-signal detection and temporal metadata.
    """

    X = ctx.series.X

    if len(X) == 0:
        raise ValueError("The loaded dataset is empty.")

    if sensor.strip().lower() in ("all", "*", "any", "none"):
        sensor = ""  # models often pass "All" for "every sensor"

    if sensor and sensor not in X.columns:
        raise ValueError(
            f"Unknown sensor '{sensor}'. Available: {list(X.columns)}"
        )

    idx_info = _timestamp_info(X.index)
    out = {
        "file": ctx.series.file_id,
        "n": len(X),
        "start": str(X.index[0]),
        "end": str(X.index[-1]),
        "sampling": idx_info.get("sampling_interval"),
    }
    if idx_info["duplicate_timestamps"] or not idx_info["monotonic"]:
        out["index_problems"] = {"duplicates": idx_info["duplicate_timestamps"],
                                 "monotonic": idx_info["monotonic"]}

    columns = [sensor] if sensor else list(X.columns)
    sensors, flags = {}, {}

    for col in columns:
        s = X[col].astype(float)
        valid = s.dropna()
        n = len(valid)

        if n == 0:
            flags[col] = ["no valid observations"]
            continue

        std = float(valid.std())
        q25, med, q75 = (float(valid.quantile(q)) for q in (0.25, 0.5, 0.75))
        # Compact core: where it sits, how much it varies, how far the extremes reach.
        sensors[col] = {"med": _sig(med), "std": _sig(std), "iqr": _sig(q75 - q25),
                        "min": _sig(valid.min()), "max": _sig(valid.max())}

        # Flags: only things that are abnormal enough to change the analysis.
        f = []
        if valid.nunique() == 1:
            f.append("constant")
        elif valid.nunique() < 20:
            f.append(f"only {valid.nunique()} distinct values (quantised)")
        if s.isna().any():
            f.append(f"{s.isna().mean():.0%} missing")
        if (valid == 0).mean() > 0.5:
            f.append("mostly zero")
        if std > 0 and n >= 2:
            x = np.arange(len(s))[s.notna().values]
            drift = float(np.polyfit(x, valid.values, 1)[0]) * len(s) / std  # total fitted change, in std units
            if abs(drift) >= 1:
                f.append(f"drifts {drift:+.1f} std over the file")
        if f:
            flags[col] = f

    out["sensors"] = sensors
    if flags:
        out["flags"] = flags
    if sensor and sensor in sensors:  # single-sensor request: add the detail the overview leaves out
        v = X[sensor].dropna().astype(float)
        sensors[sensor].update({"mean": _sig(v.mean()), "q25": _sig(v.quantile(0.25)),
                                "q75": _sig(v.quantile(0.75)), "n_unique": int(v.nunique())})
    return out


@tool
def correlations(
    threshold: float = 0.8,
    method: str = "pearson"
) -> dict:
    """Find highly correlated sensor pairs.

    Args:
        threshold: Absolute correlation threshold (0 to 1).
        method: Correlation method: pearson or spearman.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    if method not in ("pearson", "spearman"):
        raise ValueError("method must be pearson or spearman")

    X = ctx.series.X.select_dtypes(include=np.number)

    if X.shape[1] < 2:
        return {"pairs": [], "message": "At least two numeric sensors required."}

    pairs = []

    for i, a in enumerate(X.columns):
        for b in X.columns[i + 1:]:
            paired = X[[a, b]].dropna()
            n = len(paired)

            if n < 3:
                continue

            # Constant signals have undefined correlation
            if paired[a].nunique() < 2 or paired[b].nunique() < 2:
                continue

            corr = paired[a].corr(paired[b], method=method)

            if pd.notna(corr) and abs(corr) >= threshold:
                pairs.append({
                    "sensor_1": a,
                    "sensor_2": b,
                    "correlation": round(float(corr), 4),
                    "paired_observations": n
                })

    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

    return {
        "method": method,
        "threshold": threshold,
        "pairs": pairs,
        "interpretation_note": (
            "Correlation indicates association, not causation. "
            "Shared trends and autocorrelation can inflate correlations."
        )
    }


@tool
def check_stationarity(
    sensor: str,
    test: str = "both"
) -> dict:
    """Check stationarity using ADF and optionally KPSS.

    Args:
        sensor: Sensor column name.
        test: adf, kpss, or both.
    """

    from statsmodels.tsa.stattools import adfuller, kpss

    if test not in ("adf", "kpss", "both"):
        raise ValueError("test must be adf, kpss, or both")

    s = _sensor(sensor)
    n_original = len(s)
    s = s.dropna()
    n = len(s)

    if n < 20:
        raise ValueError(
            f"Insufficient observations: {n}. "
            "At least 20 valid observations are required."
        )

    if s.nunique() < 2:
        return {
            "sensor": sensor,
            "error": "Constant signal: stationarity tests are not informative.",
            "valid_observations": n,
            "missing_removed": n_original - n
        }

    results = {}

    if test in ("adf", "both"):
        try:
            stat, p, *_ = adfuller(s.values, autolag="AIC")
            results["adf"] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "reject_unit_root": bool(p < 0.05)
            }
        except Exception as e:
            results["adf"] = {"error": str(e)}

    if test in ("kpss", "both"):
        try:
            stat, p, *_ = kpss(
                s.values,
                regression="c",
                nlags="auto"
            )
            results["kpss"] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "reject_stationarity": bool(p < 0.05)
            }
        except Exception as e:
            results["kpss"] = {"error": str(e)}

    return {
        "sensor": sensor,
        "valid_observations": n,
        "missing_removed": n_original - n,
        "alpha": 0.05,
        "tests": results,
        "interpretation_note": (
            "ADF tests for a unit root; KPSS tests stationarity. "
            "Results are statistical evidence, not definitive proof."
        )
    }
