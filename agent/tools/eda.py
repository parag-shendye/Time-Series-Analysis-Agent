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

    out = {
        "file": ctx.series.file_id,
        "n_observations": len(X),
        "n_sensors": len(X.columns),
        "start": str(X.index[0]),
        "end": str(X.index[-1]),
        "time_info": _timestamp_info(X.index),
        "sensors": {}
    }

    columns = [sensor] if sensor else list(X.columns)

    for col in columns:
        s = X[col].astype(float)
        valid = s.dropna()
        n = len(valid)

        if n == 0:
            out["sensors"][col] = {
                "valid_observations": 0,
                "missing": len(s),
                "error": "No valid observations"
            }
            continue

        # Trend based only on valid observations
        slope = None
        if n >= 2 and valid.nunique() > 1:
            x = np.arange(len(s))[s.notna().values]
            slope = float(np.polyfit(x, valid.values, 1)[0])

        out["sensors"][col] = {
            "mean": round(float(valid.mean()), 4),
            "std": round(float(valid.std()), 4),
            "min": round(float(valid.min()), 4),
            "max": round(float(valid.max()), 4),
            "median": round(float(valid.median()), 4),
            "q25": round(float(valid.quantile(0.25)), 4),
            "q75": round(float(valid.quantile(0.75)), 4),
            "missing": int(s.isna().sum()),
            "missing_fraction": round(float(s.isna().mean()), 4),
            "valid_observations": n,
            "n_unique": int(valid.nunique()),
            "is_constant": bool(valid.nunique() == 1),
            "zero_fraction": round(float((valid == 0).mean()), 4),
            "trend_slope_per_observation": (
                round(slope, 6) if slope is not None else None
            )
        }

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
