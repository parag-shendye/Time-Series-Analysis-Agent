import asyncio
import atexit
import json
import os
import sys
import threading

from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

from .config import ROOT
from .db import store
from .tools import registry

MCP_URL = os.environ.get("DETECTOR_MCP_URL", "")
CONNECT_TIMEOUT_S = 120
CALL_TIMEOUT_S = 300


class _Bridge:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._client: Client | None = None
        self._error: BaseException | None = None
        self.tools: list = []
        threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-bridge").start()
        asyncio.run_coroutine_threadsafe(self._main(), self._loop)
        if not self._ready.wait(timeout=CONNECT_TIMEOUT_S):
            raise RuntimeError("timed out connecting to the detector MCP server")
        if self._error:
            raise RuntimeError(f"could not connect to the detector MCP server: {self._error!r}") from self._error

    def _target(self):
        if MCP_URL:
            return MCP_URL
        return StdioServerParameters(command=sys.executable, args=["-m", "detector_server"],
                                     env={**os.environ}, cwd=str(ROOT))

    async def _main(self):
        self._stop = asyncio.Event()
        try:
            async with Client(self._target()) as client:
                self._client = client
                self.tools = (await client.list_tools()).tools
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # surface connection failures to the constructor
            self._error = e
            self._ready.set()

    def call(self, name: str, args: dict) -> str:
        async def _call():
            r = await self._client.call_tool(name, args)
            return "".join(getattr(c, "text", "") for c in r.content), r.is_error
        text, is_error = asyncio.run_coroutine_threadsafe(_call(), self._loop).result(CALL_TIMEOUT_S)
        return f"error: {text}" if is_error else text

    def close(self):
        if self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)


_bridge: _Bridge | None = None


def _model_schema(tool) -> dict:
    """OpenAI-style schema for the model, with `file_id` removed (the agent injects it)."""
    schema = json.loads(json.dumps(tool.input_schema))
    schema.get("properties", {}).pop("file_id", None)
    schema["required"] = [r for r in schema.get("required", []) if r != "file_id"]
    return {"type": "function", "function": {"name": tool.name, "description": tool.description or "",
                                              "parameters": schema}}


def _make_caller(name: str, takes_file: bool):
    def call(**args) -> str:
        ctx = registry.ctx
        if takes_file:
            args = {**args, "file_id": ctx.series.file_id}
        out = _bridge.call(name, args)
        if name == "run_detector" and not out.startswith("error:") and ctx.con is not None:
            r = json.loads(out)  # keep a local provenance record of what the remote detector returned
            store.save_model_run(ctx.con, r["file_id"], r["detector"], r["params"], r["intervals"], run_id=ctx.run_id)
        return out
    return call


def connect() -> None:
    """Connect once per process and register the server's tools. Safe to call repeatedly."""
    global _bridge
    if _bridge is not None:
        return
    _bridge = _Bridge()
    atexit.register(_bridge.close)
    for t in _bridge.tools:
        takes_file = "file_id" in t.input_schema.get("properties", {})
        registry.register_remote(t.name, _model_schema(t), _make_caller(t.name, takes_file))
