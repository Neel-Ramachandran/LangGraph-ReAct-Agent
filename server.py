# ============================================================
#  server.py
#  Runs the ReAct agent (from agent.py) as a web API (FastAPI).
#
#  POST a question and get back the full ReAct trace (thoughts,
#  tool calls, observations, final answer) as JSON.
#
#  Setup:
#    pip install -r requirements.txt
#    # put ANTHROPIC_API_KEY=... in a .env file in this folder
#
#  Run (either works — the FastAPI instance is named `app`):
#    python server.py
#    uvicorn server:app --reload --port 8000
#
#  Then test it:
#    curl -X POST http://localhost:8000/ask \
#         -H "Content-Type: application/json" \
#         -d '{"question": "What is the cube root of 2048383?"}'
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langgraph.errors import GraphRecursionError

from agent import run_events, RECURSION_LIMIT


app = FastAPI(title="ReAct Agent API")


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    """POST a question, get back the full ReAct trace."""
    try:
        # run_events already handles multi-tool turns correctly, so the
        # trace here matches exactly what the CLI prints.
        trace = list(run_events(query.question))
    except GraphRecursionError:
        return {
            "question": query.question,
            "answer": None,
            "trace": [],
            "error": (
                f"Hit the {RECURSION_LIMIT}-step limit without finishing — "
                f"the agent may be stuck in a loop."
            ),
        }

    # The final answer is the last "answer" event.
    answer = next(
        (e["content"] for e in reversed(trace) if e["type"] == "answer"),
        None,
    )

    return {"question": query.question, "answer": answer, "trace": trace}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── RUN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # host="0.0.0.0" makes it reachable from outside the machine.
    uvicorn.run(app, host="0.0.0.0", port=8000)
