import inspect
import json
from dataclasses import dataclass, field
from typing import Callable, get_type_hints

from ..data import Series

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Context:
    series: Series | None = None
    con: object = None
    run_id: int | None = None
    with_memory: bool = True
    cache: dict = field(default_factory=dict)


ctx = Context()
TOOLS: dict[str, Callable] = {}
REMOTE: dict[str, tuple[dict, Callable]] = {}  # tools served over MCP: name -> (model schema, caller)


def tool(fn: Callable) -> Callable:
    TOOLS[fn.__name__] = fn
    return fn


def _schema(fn: Callable) -> dict:
    hints, props, required = get_type_hints(fn), {}, []
    desc, *arg_part = (inspect.getdoc(fn) or "").split("\nArgs:")
    arg_docs = {}
    for line in (arg_part[0].splitlines() if arg_part else []):
        if ":" in line:
            k, v = line.split(":", 1)
            arg_docs[k.strip()] = v.strip()
    for name, p in inspect.signature(fn).parameters.items():
        props[name] = {"type": _JSON_TYPES.get(hints.get(name, str), "string"),
                       "description": arg_docs.get(name, "")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "function", "function": {
        "name": fn.__name__, "description": desc.strip(),
        "parameters": {"type": "object", "properties": props, "required": required}}}


def register_remote(name: str, schema: dict, caller: Callable) -> None:
    """Register a tool implemented elsewhere (an MCP server). `schema` is the OpenAI-style tool schema."""
    REMOTE[name] = (schema, caller)


def schemas(names: list[str] | None = None) -> list[dict]:
    local = [_schema(f) for n, f in TOOLS.items() if names is None or n in names]
    remote = [sch for n, (sch, _) in REMOTE.items() if names is None or n in names]
    return local + remote


def call(name: str, args: dict | None) -> str:
    """Run a tool; always return a string (errors included) so the model can recover."""
    fn = TOOLS.get(name) or (REMOTE[name][1] if name in REMOTE else None)
    if fn is None:
        return f"error: unknown tool '{name}'. Available: {', '.join([*TOOLS, *REMOTE])}"
    try:
        out = fn(**(args or {}))
    except TypeError as e:
        return f"error: bad arguments for {name}: {e}"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"
    return out if isinstance(out, str) else json.dumps(out, default=str, separators=(",", ":"))
