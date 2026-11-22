"""Run the detector MCP server.

    python -m detector_server                       # stdio (the agent spawns this itself)
    python -m detector_server --http --port 8000    # streamable HTTP at http://HOST:8000/mcp

Hosting elsewhere: copy this repo (skab_detect + detector_server) and the SKAB data, set SKAB_DATA, run with
--http --host 0.0.0.0 behind TLS and authentication (this server has none: do not expose it bare), then point
the agent at it with DETECTOR_MCP_URL=https://your-host/mcp.
"""
import argparse
import asyncio

import mcp.server.stdio

from .server import build_server


async def _stdio() -> None:
    server = build_server()
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    ap = argparse.ArgumentParser(prog="detector_server")
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if a.http:
        import uvicorn
        uvicorn.run(build_server().streamable_http_app(host=a.host), host=a.host, port=a.port)
    else:
        asyncio.run(_stdio())


if __name__ == "__main__":
    main()
