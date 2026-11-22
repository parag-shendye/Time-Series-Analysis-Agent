import json
import time
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import data, mcp_client, tools
from .config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_THINK
from .db import store
from .prompts import system_prompt


MAX_RESULT_CHARS = 4096  # keep tool output small for small models
MAX_CALLS_PER_TOOL = 3   # models that loop on one tool get cut off here


@dataclass
class RunResult:
    run_id: int
    file_id: str
    answer: str
    steps: int
    tool_calls: int
    seconds: float
    trace: list = field(default_factory=list)


def _truncate(s: str) -> str:
    return s if len(s) <= MAX_RESULT_CHARS else s[:MAX_RESULT_CHARS] + "...[truncated]"


def _next_step_hint(per_tool: dict[str, int]) -> str:
    """Name the next unfinished workflow step so a stuck model has something concrete to do."""
    if "run_detector" not in per_tool:
        return "Your next step is to call run_detector (default level_shift) to get candidate anomaly intervals."
    if "extract_window_features" not in per_tool:
        return "Your next step is to call extract_window_features on a candidate interval from run_detector."
    if "save_label" not in per_tool:
        return "Your next step is to call save_label for each supported anomaly, then give your final answer."
    return "Give your final answer now."


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step: int                  # model turns taken
    n_calls: int               # tool calls requested (incl. rejected duplicates)
    bad_turns: int             # times we nudged a model that answered without using tools
    seen: dict[str, bool]      # "tool:args" pairs already executed
    per_tool: dict[str, int]


def build_graph(model: str = OLLAMA_MODEL, max_steps: int = 15):
    llm = ChatOllama(model=model, base_url=OLLAMA_HOST, temperature=0, num_ctx=8192,
                     reasoning=OLLAMA_THINK).bind_tools(tools.schemas())

    def model_node(state: AgentState) -> dict:
        step = state["step"] + 1
        out: list[BaseMessage] = []
        if step == max_steps - 2:  # budget nearly gone: force wrap-up
            out.append(HumanMessage(
                "Only 3 steps left. Do not gather more data. Call save_label for each anomaly you believe in, "
                "then give your final answer."))
        reply = llm.invoke(state["messages"] + out)
        return {"messages": out + [reply], "step": step}

    def tools_node(state: AgentState) -> dict:
        seen, per_tool, n_calls = dict(state["seen"]), dict(state["per_tool"]), state["n_calls"]
        results = []
        for tc in state["messages"][-1].tool_calls:
            name = tc["name"]
            args = {k: v for k, v in (tc["args"] or {}).items() if v not in ("", None)}  # "" == omitted
            key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            if key in seen:
                result = (f"error: you already called {name} with these exact arguments; its result is earlier in "
                          "the conversation. Do not repeat it. " + _next_step_hint(per_tool))
            elif per_tool.get(name, 0) >= MAX_CALLS_PER_TOOL:
                result = (f"error: {name} has been called {MAX_CALLS_PER_TOOL} times already. "
                          "Use the results you have. " + _next_step_hint(per_tool))
            else:
                seen[key] = True
                per_tool[name] = per_tool.get(name, 0) + 1
                result = _truncate(tools.call(name, args))
            n_calls += 1
            results.append(ToolMessage(content=result, name=name, tool_call_id=tc["id"]))
        return {"messages": results, "seen": seen, "per_tool": per_tool, "n_calls": n_calls}

    def nudge_node(state: AgentState) -> dict:
        hint = "Start with describe_series." if state["n_calls"] == 0 else _next_step_hint(state["per_tool"])
        return {"messages": [HumanMessage(f"You have not finished the investigation. Do not narrate; call a tool. {hint}")],
                "bad_turns": state["bad_turns"] + 1}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if state["step"] >= max_steps and last.tool_calls:
            return END  # budget exhausted mid-investigation
        if last.tool_calls:
            return "tools"
        unfinished = state["n_calls"] == 0 or "run_detector" not in state["per_tool"]
        if unfinished and state["bad_turns"] < 3:  # answered before detecting anything: push it back
            return "nudge"
        return END

    g = StateGraph(AgentState)
    g.add_node("model", model_node)
    g.add_node("tools", tools_node)
    g.add_node("nudge", nudge_node)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", route, {"tools": "tools", "nudge": "nudge", END: END})
    g.add_edge("tools", "model")
    g.add_edge("nudge", "model")
    return g.compile()


def run(file_id: str, model: str = OLLAMA_MODEL, max_steps: int = 15, with_memory: bool = True,
        verbose: bool = True, con=None) -> RunResult:
    con = con or store.connect()
    series = data.load(file_id)
    store.register_dataset(con, series)
    run_id = store.start_run(con, file_id, model, with_memory)
    tools.ctx.series, tools.ctx.con, tools.ctx.run_id, tools.ctx.with_memory = series, con, run_id, with_memory

    mcp_client.connect()  # detectors live on the MCP server: register them before binding tools
    graph = build_graph(model, max_steps)
    state: AgentState = {
        "messages": [SystemMessage(system_prompt(with_memory)),
                     HumanMessage(f"Analyse file {file_id} and find its anomalies.")],
        "step": 0, "n_calls": 0, "bad_turns": 0, "seen": {}, "per_tool": {}}

    t0, trace, final = time.time(), [], state
    # recursion_limit is only a backstop; max_steps is enforced in `route`
    for final in graph.stream(state, {"recursion_limit": 4 * max_steps + 10}, stream_mode="values"):
        msg = final["messages"][-1]
        if isinstance(msg, ToolMessage) and verbose:
            print(f"[{final['step']}] {msg.name} -> {msg.content[:200]}")

    msgs = final["messages"]
    calls = {}  # tool_call_id -> args, to pair with results in the trace
    for m in msgs:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                calls[tc["id"]] = tc["args"]
        elif isinstance(m, ToolMessage):
            trace.append({"tool": m.name, "args": calls.get(m.tool_call_id), "result": m.content})
    last = msgs[-1]
    if isinstance(last, AIMessage) and not last.tool_calls:
        answer = last.text
        trace.append({"event": "final", "content": answer})
    else:
        answer = "(step budget exhausted without a final answer)"
        trace.append({"event": "budget_exhausted"})

    store.finish_run(con, run_id, trace)
    return RunResult(run_id, file_id, answer, final["step"], final["n_calls"], time.time() - t0, trace)
