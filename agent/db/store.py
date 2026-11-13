import json
import sqlite3
from pathlib import Path

import pandas as pd

from ..config import DB_PATH
from ..data import Series

SCHEMA = Path(__file__).with_name("schema.sql")


def connect(path=DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA.read_text())
    return con


def register_dataset(con, s: Series, split: str | None = None):
    con.execute(
        "INSERT OR REPLACE INTO datasets VALUES (?,?,?,?,?,?)",
        (s.file_id, s.group, str(s.X.index[0]), str(s.X.index[-1]), len(s.X), split),
    )


def _intervals(mask: pd.Series) -> list[tuple[str, str]]:
    m = mask.astype(bool)
    grp = (m != m.shift()).cumsum()
    return [(str(g.index[0]), str(g.index[-1])) for _, g in m[m].groupby(grp[m])]


def seed_ground_truth(con, s: Series):
    """Store truth anomaly intervals and changepoint timestamps as ground_truth labels."""
    if s.truth is None:
        return
    con.execute("DELETE FROM labels WHERE file_id=? AND source='ground_truth'", (s.file_id,))
    for a, b in _intervals(s.truth["anomaly"]):
        con.execute("INSERT INTO labels(file_id,start_ts,end_ts,label,source,confidence) "
                    "VALUES (?,?,?,'anomaly','ground_truth',1.0)", (s.file_id, a, b))
    if "changepoint" in s.truth:
        for a, b in _intervals(s.truth["changepoint"]):
            con.execute("INSERT INTO labels(file_id,start_ts,end_ts,label,source,confidence) "
                        "VALUES (?,?,?,'changepoint','ground_truth',1.0)", (s.file_id, a, b))


def save_pattern(con, name, description, signature=None, sensors=None, file_id=None,
                 start_ts=None, end_ts=None, confidence=None, created_by="agent") -> int:
    cur = con.execute(
        "INSERT INTO patterns(name,description,signature_json,sensors,file_id,start_ts,end_ts,"
        "confidence,created_by) VALUES (?,?,?,?,?,?,?,?,?)",
        (name, description, json.dumps(signature) if signature else None,
         ",".join(sensors) if sensors else None, file_id, start_ts, end_ts, confidence, created_by),
    )
    con.commit()
    return cur.lastrowid


def add_label(con, file_id, start_ts, end_ts, label, source="agent", confidence=None,
              pattern_id=None, run_id=None) -> int:
    cur = con.execute(
        "INSERT INTO labels(file_id,start_ts,end_ts,label,source,confidence,pattern_id,run_id) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (file_id, start_ts, end_ts, label, source, confidence, pattern_id, run_id),
    )
    con.commit()
    return cur.lastrowid


def get_labels(con, file_id, source=None) -> list[dict]:
    q, args = "SELECT * FROM labels WHERE file_id=?", [file_id]
    if source:
        q += " AND source=?"
        args.append(source)
    return [dict(r) for r in con.execute(q + " ORDER BY start_ts", args)]


def save_profile(con, file_id, sensor, stats: dict):
    con.execute("INSERT OR REPLACE INTO profiles(file_id,sensor,stats_json) VALUES (?,?,?)",
                (file_id, sensor, json.dumps(stats)))
    con.commit()


def get_patterns(con, status=None) -> list[dict]:
    q, args = "SELECT * FROM patterns", []
    if status:
        q += " WHERE status=?"
        args.append(status)
    return [dict(r) for r in con.execute(q + " ORDER BY id", args)]


def set_pattern_status(con, pattern_id, status):
    con.execute("UPDATE patterns SET status=? WHERE id=?", (status, pattern_id))
    con.commit()


def find_similar_patterns(con, sensors=None, text=None, limit=5) -> list[dict]:
    """Rank non-rejected patterns by sensor overlap and keyword overlap with `text`."""
    want = set(sensors or [])
    words = {w for w in (text or "").lower().split() if len(w) > 3}
    scored = []
    for p in get_patterns(con):
        if p["status"] == "rejected":
            continue
        have = set((p["sensors"] or "").split(","))
        hay = f"{p['name']} {p['description']}".lower()
        score = len(want & have) / max(len(want | have), 1) + sum(w in hay for w in words) * 0.25
        if p["status"] == "confirmed":
            score += 0.25
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda t: -t[0])
    return [{**p, "score": round(s, 2)} for s, p in scored[:limit]]


def save_model_run(con, file_id, detector, params: dict, intervals, metrics=None, run_id=None):
    con.execute(
        "INSERT OR REPLACE INTO model_runs(file_id,detector,params_json,intervals_json,metrics_json,run_id) "
        "VALUES (?,?,?,?,?,?)",
        (file_id, detector, json.dumps(params, sort_keys=True), json.dumps(intervals),
         json.dumps(metrics) if metrics else None, run_id))
    con.commit()


def get_cached_model_run(con, file_id, detector, params: dict):
    r = con.execute("SELECT intervals_json FROM model_runs WHERE file_id=? AND detector=? AND params_json=?",
                    (file_id, detector, json.dumps(params, sort_keys=True))).fetchone()
    return json.loads(r[0]) if r else None


def start_run(con, file_id, llm_model, with_memory=True) -> int:
    cur = con.execute("INSERT INTO runs(file_id,llm_model,with_memory) VALUES (?,?,?)",
                      (file_id, llm_model, int(with_memory)))
    con.commit()
    return cur.lastrowid


def finish_run(con, run_id, trace: list):
    con.execute("UPDATE runs SET finished_at=CURRENT_TIMESTAMP, trace_json=? WHERE id=?",
                (json.dumps(trace, default=str), run_id))
    con.commit()


def add_note(con, note, file_id=None, run_id=None) -> int:
    cur = con.execute("INSERT INTO agent_notes(file_id,run_id,note) VALUES (?,?,?)", (file_id, run_id, note))
    con.commit()
    return cur.lastrowid


def get_notes(con, file_id=None, limit=10) -> list[dict]:
    q, args = "SELECT * FROM agent_notes", []
    if file_id:
        q += " WHERE file_id=?"
        args.append(file_id)
    return [dict(r) for r in con.execute(q + " ORDER BY id DESC LIMIT ?", args + [limit])]
