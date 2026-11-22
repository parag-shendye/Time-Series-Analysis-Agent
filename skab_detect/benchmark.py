"""Score detectors against SKAB ground truth."""
from pathlib import Path
from typing import Iterable

import pandas as pd

from .detectors import detect
from .metrics import point_scores
from .skab import list_files, load, split


def evaluate(root: Path | str, files: Iterable[str], detectors: Iterable[str],
             params: dict | None = None) -> pd.DataFrame:
    """One row per (file, detector) with precision/recall/F1. Files without an anomaly column are skipped."""
    rows = []
    for fid in files:
        X, truth = load(root, fid)
        if truth is None or "anomaly" not in truth:
            continue
        t = truth["anomaly"].astype(bool)
        for name in detectors:
            pred = detect(X, name, **(params or {}))
            rows.append({"file": fid, "detector": name, "n_truth": int(t.sum()), "n_pred": int(pred.sum()),
                         **point_scores(pred, t)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/min F1 and mean precision/recall per detector."""
    g = df.groupby("detector")
    return pd.DataFrame({"n_files": g.size(), "f1_mean": g.f1.mean(), "f1_min": g.f1.min(),
                         "precision": g.precision.mean(), "recall": g.recall.mean()}).round(3)


__all__ = ["evaluate", "summarize", "list_files", "split"]
