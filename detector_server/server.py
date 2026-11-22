"""MCP server exposing the SKAB anomaly detectors (low-level `mcp` SDK, no FastMCP).

The server owns the data: tools take a `file_id` (e.g. "valve1/3"), load the sensor frame from SKAB_DATA,
run a detector from `skab_detect`, and return intervals. Ground truth is never read, so a hosted server
cannot leak labels to a client. Detector results are cached in memory by (file, detector, params).
"""
import asyncio
import json
import os
from collections import OrderedDict
from pathlib import Path

import mcp_types as types
from mcp.server.lowlevel import Server

from skab_detect import DETECTORS, detect, mask_to_intervals, skab

ROOT = Path(__file__).resolve().parent.parent
SKAB_DATA = Path(os.environ.get("SKAB_DATA", ROOT.parent / "SKAB" / "data"))
CACHE_SIZE = 256

DETECTOR_HELP = {
    "level_shift": "Sustained level shifts. Finds more faults but also flags slow drift on fault-free data. Default.",
    "rolling_mean": "A significant change in a sensor's rolling mean that later reverts. Far fewer false alarms but misses weak faults.",
    "isolation_forest": "Isolated outlier points (contamination sets the expected fraction). Weak on SKAB's sustained faults.",
}

TOOLS = [
    types.Tool(
        name="list_detectors",
        description="List the available anomaly detectors and when to use each.",
        input_schema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="run_detector",
        description=("Run an anomaly detector over all sensors of one file and return the predicted anomaly "
                     "intervals (timestamps). Treat them as hypotheses, not confirmed faults. "
                     + " ".join(f"{k}: {v}" for k, v in DETECTOR_HELP.items())),
        input_schema={
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Dataset file, e.g. valve1/3."},
                "detector": {"type": "string", "enum": list(DETECTORS), "default": "level_shift",
                             "description": "Detector name from list_detectors."},
                "contamination": {"type": "number", "minimum": 0.01, "maximum": 0.4, "default": 0.1,
                                  "description": "Only used by isolation_forest: expected fraction of anomalous samples."},
            },
            "required": ["file_id"],
        },
    ),
]

_cache: "OrderedDict[tuple, list]" = OrderedDict()


def _load_sensors(file_id: str):
    if file_id not in set(skab.list_files(SKAB_DATA)):  # also blocks path traversal
        raise ValueError(f"unknown file_id '{file_id}'")
    X, _truth = skab.load(SKAB_DATA, file_id)
    return X  # ground truth deliberately dropped here


def _run_detector(file_id: str, detector: str = "level_shift", contamination: float = 0.1) -> dict:
    if detector not in DETECTORS:
        raise ValueError(f"unknown detector '{detector}'; choose from {list(DETECTORS)}")
    params = {"contamination": round(min(max(float(contamination), 0.01), 0.4), 3)}
    key = (file_id, detector, tuple(sorted(params.items())))
    if key in _cache:
        _cache.move_to_end(key)
        intervals, cached = _cache[key], True
    else:
        X = _load_sensors(file_id)
        intervals = [[str(a), str(b)] for a, b in mask_to_intervals(detect(X, detector, **params))]
        _cache[key] = intervals
        if len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
        cached = False
    return {"file_id": file_id, "detector": detector, "params": params, "cached": cached,
            "n_intervals": len(intervals), "intervals": intervals[:15]}


def _result(payload: dict, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=json.dumps(payload, separators=(",", ":")))],
                                structured_content=None if is_error else payload, is_error=is_error)


async def on_list_tools(ctx, params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    args = params.arguments or {}
    try:
        if params.name == "list_detectors":
            return _result({"detectors": [{"name": n, "description": DETECTOR_HELP.get(n, "")} for n in DETECTORS]})
        if params.name == "run_detector":
            return _result(await asyncio.to_thread(_run_detector, **args))  # detectors are CPU-bound
        return _result({"error": f"unknown tool '{params.name}'"}, is_error=True)
    except (ValueError, TypeError) as e:  # bad arguments: report to the model, don't crash the server
        return _result({"error": f"{type(e).__name__}: {e}"}, is_error=True)


def build_server() -> Server:
    return Server("skab-detectors", version="0.1.0",
                  instructions="Anomaly detectors for the SKAB water-circulation dataset.",
                  on_list_tools=on_list_tools, on_call_tool=on_call_tool)
