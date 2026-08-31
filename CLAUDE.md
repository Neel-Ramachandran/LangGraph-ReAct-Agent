# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal ReAct (Reason → Act → Observe) agent built on LangGraph + `langchain-anthropic`,
exposed two ways: as a CLI script and as a FastAPI service. The agent itself lives in one shared
module; the two entry points only differ in how they present the trace.

## Commands

Setup (no venv is checked in; `requirements.txt` is unpinned):

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use source .venv/bin/activate on POSIX
pip install -r requirements.txt
```

`ANTHROPIC_API_KEY` is read from `.env` in the repo root via `load_dotenv()` (called in `agent.py`).

Run the CLI agent (executes the hardcoded example question in the `__main__` block):

```bash
python react_agent.py
```

Run the API server:

```bash
python server.py                          # uvicorn on 0.0.0.0:8000
uvicorn server:app --reload --port 8000   # auto-reload during development
```

Exercise the API:

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the cube root of 2048383?"}'
curl http://localhost:8000/health
```

There is no test suite, linter, formatter, or CI configuration in this repo.

## Architecture

Three files, one agent:

- **`agent.py`** — the entire agent: state, tools, the LLM, the graph, and `run_events()`. This is the
  single source of truth. `react_agent.py` and `server.py` both import from it and contain no agent logic
  of their own, so the CLI and the API can never drift apart.
- **`react_agent.py`** — prints the trace to the terminal (`[THOUGHT]/[OBSERVE]/[ANSWER]`).
- **`server.py`** — returns the same trace as JSON over HTTP.

The graph:

```
entry → agent_node ──should_continue()──> "tools" → tool_node ──┐
                     │                                          │
                     └─> "end" → END                            │
                          ▲                                     │
                          └─────────────────────────────────────┘
```

- **State** is a `TypedDict` with a single `messages` key annotated `Annotated[list[BaseMessage], operator.add]`.
  That reducer is what makes the loop work — every node returns `{"messages": [msg]}` and LangGraph
  *appends* rather than replaces. Returning a full list here would clobber history.
- **`agent_node`** is the Reason step: one `llm_with_tools.invoke(...)` call.
- **`should_continue`** is the only routing logic: if the last message has non-empty `tool_calls`, go to
  `"tools"`; otherwise `"end"`. The loop terminates when Claude declines to call a tool.
- **`tool_node`** (prebuilt `ToolNode`) is Act + Observe; the unconditional `tools → agent` edge closes the loop.

### `run_events()` — the trace walker (read this before touching trace code)

`run_events(question)` is the shared generator both entry points consume. It yields normalised events:

```python
{"type": "thought",     "tool": <name>, "input": <args>}
{"type": "observation", "tool": <name>, "result": <str>}
{"type": "answer",      "content": <str>}
```

It streams the graph with `stream_mode="values"` (each step yields the *full* accumulated state) and
walks **only the new messages** each step via `messages[seen:]`. This is load-bearing: when Claude calls
several tools in one turn, `ToolNode` appends *all* their results in a single step. The earlier code read
only `messages[-1]` and silently dropped every observation but the last. Any change here must preserve the
`messages[seen:]` slice, or multi-tool turns will lose observations again.

## Model and config

- The model id lives in one place: `MODEL = "claude-fable-5"` in `agent.py`. Change it there and both
  entry points pick it up. Consult the `/claude-api` skill before swapping model ids.
- **Fable specifics that matter here:** thinking is always on, so a final `AIMessage` carries a list of
  content blocks (thinking + text), not a bare string — `_message_text()` extracts just the text blocks.
  `langchain-anthropic` sends only `model`/`max_tokens`/`messages` (no `temperature`/`top_p`/`thinking`
  config), so Fable's stricter 400s aren't triggered. `max_tokens` resolves to the langchain fallback
  (4096) since there's no bundled model profile for Fable.
- `RECURSION_LIMIT = 25` (in `agent.py`) caps the Reason→Act loop. On overflow LangGraph raises
  `GraphRecursionError`; both entry points catch it (the CLI prints `[STOPPED]`, the API returns an
  `error` field with `answer: null`).
- `load_dotenv()` is called *before* the `langchain_anthropic` import in `agent.py`. Keep that ordering —
  the client reads the key at import time.

## Other things worth knowing

- `calculator` runs `eval()` with `{"__builtins__": {}}` and `math` as the only global. That blocks the
  obvious attacks but is **not** a real sandbox (`().__class__.__mro__` traversal is still reachable), and
  the expression string comes straight from model output. Treat it as demo-grade; harden it
  (`ast.literal_eval` + an operator allowlist, or `simpleeval`) before exposing to untrusted input.
- `server.py` binds `0.0.0.0` with no auth, no CORS config, and no rate limiting.
- This directory is not a git repository. Note `.env` holds a live `ANTHROPIC_API_KEY` — add a `.gitignore`
  excluding it before any `git init`/first commit.
