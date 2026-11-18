CREATE TABLE IF NOT EXISTS datasets (
    file_id TEXT PRIMARY KEY,          -- e.g. valve1/3
    grp TEXT NOT NULL,
    start_ts TEXT, end_ts TEXT,
    n_samples INTEGER,
    split TEXT CHECK (split IN ('train','test'))
);

CREATE TABLE IF NOT EXISTS profiles (
    file_id TEXT NOT NULL REFERENCES datasets(file_id),
    sensor TEXT NOT NULL,
    stats_json TEXT NOT NULL,          -- mean/std/trend/stationarity/etc.
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (file_id, sensor)
);

CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    signature_json TEXT,               -- feature vector / per-sensor deltas
    sensors TEXT,                      -- comma-separated sensors involved
    file_id TEXT REFERENCES datasets(file_id),
    start_ts TEXT, end_ts TEXT,
    confidence REAL,
    created_by TEXT NOT NULL CHECK (created_by IN ('agent','human','seed')),
    status TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed','confirmed','rejected')),
    embedding BLOB,                    -- optional nomic-embed-text vector of description
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL REFERENCES datasets(file_id),
    start_ts TEXT NOT NULL, end_ts TEXT NOT NULL,
    label TEXT NOT NULL,               -- normal | anomaly | changepoint | <pattern name>
    source TEXT NOT NULL CHECK (source IN ('ground_truth','model','agent','human')),
    confidence REAL,
    pattern_id INTEGER REFERENCES patterns(id),
    run_id INTEGER REFERENCES runs(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_labels_file ON labels(file_id, source);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT REFERENCES datasets(file_id),
    llm_model TEXT,
    with_memory INTEGER DEFAULT 1,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    trace_json TEXT                    -- tool-call trace
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL REFERENCES datasets(file_id),
    detector TEXT NOT NULL,
    params_json TEXT NOT NULL,
    intervals_json TEXT,               -- predicted anomaly intervals
    metrics_json TEXT,                 -- f1/far/mar/... (only if scored)
    run_id INTEGER REFERENCES runs(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (file_id, detector, params_json)
);

CREATE TABLE IF NOT EXISTS agent_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT REFERENCES datasets(file_id),
    run_id INTEGER REFERENCES runs(id),
    note TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
