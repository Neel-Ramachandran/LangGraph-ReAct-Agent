# ReAct Agent

A small ReAct agent I built to understand how tool-calling agents actually work
under the hood. ReAct stands for Reason + Act: the model looks at your question,
decides whether it needs a tool, calls it, reads the result, and loops until it
can answer. This repo is a stripped-down version of that loop wired up with
LangGraph and Claude, with two tiny tools so you can watch it think.

The thing I wanted to see was the trace, not just the final answer. When you ask
it something that needs two tools at once, you can watch it fire both, read both
results, and only then write the answer.

## What ReAct is

It's a loop with three steps. Reason: the model decides what to do next. Act: it
calls a tool if it needs one. Observe: it reads what the tool returned. Then it
goes back to Reason and repeats until it has nothing left to call, at which point
it answers. LangGraph is what wires those steps into a graph and runs the loop.

## Setup

```
python -m venv .venv
.venv\Scripts\activate       # Windows; use "source .venv/bin/activate" on macOS/Linux
pip install -r requirements.txt
```

Put your Anthropic key in a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Running it

Two ways. The command-line version prints the whole trace to the terminal:

```
python react_agent.py
```

Or run it as a web API and POST questions to it:

```
python server.py
```

Then in another terminal (the `^` line continuation below is cmd.exe syntax; use `` ` `` in
PowerShell or `\` in bash/zsh):

```
curl -X POST http://localhost:8000/ask ^
     -H "Content-Type: application/json" ^
     -d "{\"question\": \"What is the cube root of 2048383?\"}"
```

You get back the answer plus the full trace as JSON: every thought, tool call,
and observation the agent went through to get there.

There's also a `GET /health` endpoint if you just want to check the server is up.

## The two tools

| Tool | Input | What it does |
|------|-------|--------------|
| `calculator` | expression | Evaluates a math expression (supports `math.sqrt`, `math.cbrt`, etc.) |
| `get_date` | none | Returns the current date and time |

They're deliberately simple. The point isn't the tools, it's watching the model
decide which one to reach for and when.

Note: `calculator` evaluates expressions with `eval()` (restricted to `math` and
no builtins). That's demo-grade, not a real sandbox — don't expose it to
untrusted input without hardening it first.

## How it's put together

The agent itself lives in one file, `agent.py`: the tools, the model, the graph,
and the loop that walks the trace. The two entry points both import from it and
only differ in how they show the result:

- `react_agent.py` prints the trace to the terminal
- `server.py` serves the same agent as a FastAPI endpoint

So there's a single source of truth for what the agent is, and adding a tool or
swapping the model only happens in one place.

The loop keeps going until the model stops asking for tools, with a 25-step
recursion limit so a misbehaving tool can't spin forever (past that, the CLI
prints `[STOPPED]` and the API returns an `error` field instead of an answer).
The interesting bit is that when the model calls more than one tool in a single
turn, both results get recorded, not just the last one.

## Built with

- Python
- LangGraph + LangChain
- Claude (via `langchain-anthropic`)
- FastAPI + Uvicorn for the web version

## Author

Neel Ramachandran
