"""CLI: python -m agent run --file valve1/3 [--model M] [--steps N] [--no-memory]"""
import argparse
import sys

from . import llm
from .config import OLLAMA_MODEL
from .graph import run


def main():
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; model output is unicode
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap =argparse.ArgumentParser(prog="agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse one SKAB file")
    r.add_argument("--file", required=True, help="e.g. valve1/3")
    r.add_argument("--model", default=OLLAMA_MODEL)
    r.add_argument("--steps", type=int, default=12)
    r.add_argument("--no-memory", action="store_true")
    sub.add_parser("check", help="verify Ollama connectivity")
    a = ap.parse_args()

    if a.cmd == "check":
        llm.check()
        print("ok")
        return
    res = run(a.file, model=a.model, max_steps=a.steps, with_memory=not a.no_memory)
    print(f"\n=== run {res.run_id}: {res.steps} steps, {res.tool_calls} tool calls, {res.seconds:.0f}s ===")
    print(res.answer)


if __name__ == "__main__":
    main()
