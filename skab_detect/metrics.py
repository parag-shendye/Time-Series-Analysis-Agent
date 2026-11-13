"""Point-wise scoring of a predicted anomaly mask against ground truth."""
import numpy as np


def point_scores(pred, truth) -> dict:
    """Precision, recall and F1 over samples. Each is 0.0 when its denominator is zero."""
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    if p.shape != t.shape:
        raise ValueError(f"length mismatch: pred {p.shape} vs truth {t.shape}")
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
