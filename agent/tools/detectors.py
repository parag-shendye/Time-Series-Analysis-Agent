from skab_detect import DETECTORS, detect, mask_to_intervals

from ..db import store
from .registry import ctx, tool


@tool
def list_detectors() -> dict:
    """List available anomaly detectors."""
    return {"detectors": list(DETECTORS)}


@tool
def run_detector(detector: str = "level_shift", contamination: float = 0.1) -> dict:
    """Run an anomaly detector over all sensors and return predicted anomaly intervals.
    level_shift (default) finds sustained level shifts: it finds more faults but also flags slow drift
    on fault-free data. rolling_mean flags a significant change in a sensor's rolling mean that later
    reverts: far fewer false alarms but it misses weak faults. isolation_forest flags isolated outlier points.

    Args:
        detector: detector name from list_detectors.
        contamination: only used by isolation_forest: expected fraction of anomalous samples, 0.01 to 0.4.
    """
    if detector not in DETECTORS:
        raise ValueError(f"unknown detector '{detector}'; choose from {list(DETECTORS)}")
    params = {"contamination": round(min(max(float(contamination), 0.01), 0.4), 3)}
    fid = ctx.series.file_id
    cached = store.get_cached_model_run(ctx.con, fid, detector, params)
    if cached is None:
        cached = [[str(a), str(b)] for a, b in mask_to_intervals(detect(ctx.series.X, detector, **params))]
        store.save_model_run(ctx.con, fid, detector, params, cached, run_id=ctx.run_id)
    return {"detector": detector, "params": params, "n_intervals": len(cached), "intervals": cached[:15]}
