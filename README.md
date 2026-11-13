# skab-agent

An LLM agent (local, via Ollama tool calling) that analyses time series from the
[SKAB](https://github.com/waico/SKAB) anomaly-detection dataset: it profiles a file, looks for
changepoints, runs anomaly detectors, labels what it finds, and stores patterns in SQLite so later
runs can reuse them.

All commands below use [uv](https://docs.astral.sh/uv/). Run them from the repo root.

## Prerequisites

1. **uv** installed.
2. **Ollama** reachable over HTTP (it can run in Docker) with a tool-calling model pulled:
   ```
   docker exec <ollama-container> ollama pull qwen3:8b
   ```
3. **SKAB data.** By default the code reads `../SKAB/data` (the sibling SKAB repo, which has the
   `valve1`, `valve2`, `other` and `anomaly-free` folders).

## Setup

```
uv sync
```
create db and register all files:

```
uv run python scripts_init_db.py
```

This writes `skab_agent.db`, registers every SKAB file with a train/test split (every 4th file is
held out), and seeds the ground-truth labels for the training files.

## Running the agent

`agent/__main__.py` is the CLI entry point. Run it as a module:

```
uv run python -m agent check
uv run python -m agent run --file valve1/3
```

### Commands

| Command | What it does |
|---|---|
| `check` | Verifies Ollama is reachable and the configured model is pulled. |
| `run --file FILE` | Runs the agent on one SKAB file and prints each tool call, then the final answer. |

Examples:

```
uv run python -m agent run --file valve1/3 --steps 15
uv run python -m agent run --file valve2/1 --model qwen3.5:2b --no-memory
```

Every run is logged to the `runs` table with its full tool-call trace, and any labels, patterns and
notes the agent saved are written to `skab_agent.db`. Plots are saved as PNGs under `artifacts/<group>_<n>/` (override with `SKAB_AGENT_ARTIFACTS`). The agent never sees the ground-truth columns.

## Configuration

Set these environment variables to override the defaults in `agent/config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is listening. |
| `OLLAMA_MODEL` | `qwen3:8b` | Default model for `run`. |
| `SKAB_DATA` | `../SKAB/data` | SKAB data folder. |
| `SKAB_AGENT_DB` | `./skab_agent.db` | SQLite database path. |