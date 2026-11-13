import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from ..config import ARTIFACTS
from .registry import ctx, tool

MAX_SENSORS = 5


def _parse_intervals(text: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    for part in filter(None, (p.strip() for p in text.split(";"))):
        a, _, b = part.partition(",")
        if not b:
            raise ValueError(f"interval '{part}' must look like 'start,end'")
        out.append((pd.Timestamp(a.strip()), pd.Timestamp(b.strip())))
    return out


def _parse_times(text: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(p.strip()) for p in text.split(";") if p.strip()]


@tool
def plot_series(sensors: str, shade_intervals: str = "", point_times: str = "", title: str = "") -> dict:
    """Save a PNG plot of sensors over time, marking anomalies. Pick the marker per anomaly: shade_intervals
    for sustained anomalies (level shifts, drifts, faults lasting several seconds), point_times for brief
    spikes or single-moment events such as a changepoint.

    Args:
        sensors: comma-separated sensor names, at most 4, e.g. Current,Pressure.
        shade_intervals: semicolon-separated 'start,end' timestamps to shade, e.g. 2020-03-09 11:16:01,2020-03-09 11:16:25;2020-03-09 11:30:00,2020-03-09 11:31:00.
        point_times: semicolon-separated timestamps to mark with a point on each sensor, e.g. 2020-03-09 11:16:09;2020-03-09 11:20:00.
        title: optional figure title.
    """
    X = ctx.series.X
    names = [s.strip() for s in sensors.split(",") if s.strip()]
    bad = [n for n in names if n not in X]
    if bad or not names:
        raise ValueError(f"unknown sensors {bad or sensors!r}; choose from {list(X.columns)}")
    names = names[:MAX_SENSORS]
    spans, points = _parse_intervals(shade_intervals), _parse_times(point_times)

    fig, axes = plt.subplots(len(names), 1, figsize=(11, 2.4 * len(names) + 0.8), sharex=True, squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        ax.plot(X.index, X[name], lw=0.9, color="tab:blue")
        ax.set_ylabel(name, fontsize=8)
        for a, b in spans:
            ax.axvspan(a, b, color="tab:red", alpha=0.25, lw=0)
        for t in points:
            nearest = X.index[X.index.get_indexer([t], method="nearest")[0]]
            ax.plot(nearest, X.loc[nearest, name], "o", color="tab:red", ms=6, mec="white", mew=0.8)
    axes[0, 0].set_title(title or f"{ctx.series.file_id}  ({len(spans)} shaded, {len(points)} points)", fontsize=10)
    fig.tight_layout()

    out_dir = ARTIFACTS / ctx.series.file_id.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(out_dir.glob(f"run{ctx.run_id}_*.png"))) + 1
    path = out_dir / f"run{ctx.run_id}_plot{n}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"saved": str(path), "sensors": names, "shaded": len(spans), "points": len(points)}
