from ..db import store
from .registry import ctx, tool


def _split(csv: str) -> list[str]:
    return [s.strip() for s in csv.split(",") if s.strip()]


@tool
def query_patterns(sensors: str = "", text: str = "") -> dict:
    """Search remembered patterns from earlier files by sensors involved and/or keywords.

    Args:
        sensors: comma-separated sensor names, e.g. Current,Pressure.
        text: keywords describing what you are looking for.
    """
    if not ctx.with_memory:
        return {"patterns": [], "note": "memory disabled"}
    found = store.find_similar_patterns(ctx.con, _split(sensors), text)
    return {"patterns": [{"id": p["id"], "name": p["name"], "description": p["description"][:200],
                          "sensors": p["sensors"], "status": p["status"], "score": p["score"]} for p in found]}


@tool
def save_pattern(name: str, description: str, sensors: str = "", start: str = "", end: str = "",
                 confidence: float = 0.5) -> dict:
    """Save a newly discovered pattern for reuse on later files. Saved as 'proposed' until confirmed.

    Args:
        name: short pattern name, e.g. current_spike_with_pressure_drop.
        description: what the pattern looks like and what it indicates.
        sensors: comma-separated sensors involved.
        start: optional example window start timestamp.
        end: optional example window end timestamp.
        confidence: 0 to 1.
    """
    pid = store.save_pattern(ctx.con, name, description, sensors=_split(sensors), file_id=ctx.series.file_id,
                             start_ts=start or None, end_ts=end or None, confidence=confidence)
    return {"saved_pattern_id": pid}


@tool
def save_label(start: str, end: str, label: str, confidence: float = 0.5) -> dict:
    """Record your judgement that a time window is an anomaly, changepoint or normal.

    Args:
        start: window start timestamp.
        end: window end timestamp.
        label: one of anomaly, changepoint, normal.
        confidence: 0 to 1.
    """
    if label not in ("anomaly", "changepoint", "normal"):
        raise ValueError("label must be anomaly, changepoint or normal")
    lid = store.add_label(ctx.con, ctx.series.file_id, start, end, label, source="agent",
                          confidence=confidence, run_id=ctx.run_id)
    return {"saved_label_id": lid}


@tool
def save_note(note: str) -> dict:
    """Save a short free-text lesson or observation about this file for future runs.

    Args:
        note: the observation.
    """
    return {"saved_note_id": store.add_note(ctx.con, note, ctx.series.file_id, ctx.run_id)}
