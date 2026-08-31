# ============================================================
#  agent.py
#  The shared ReAct (Reason -> Act -> Observe) agent.
#
#  Both entry points import from here so the CLI (react_agent.py)
#  and the web API (server.py) can never drift apart:
#    - react_agent.py  -> prints the trace to the terminal
#    - server.py       -> returns the same trace as JSON
#
#  Everything to do with WHAT the agent is (tools, model, graph)
#  lives in this file. The entry points only decide how to
#  present the trace that run_events() yields.
# ============================================================

from typing import TypedDict, Annotated
import operator
import math
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  # reads ANTHROPIC_API_KEY from a .env file  (must run before the import below)

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, BaseMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# The one place the model id is written. Change it here and both
# the CLI and the API pick it up.
MODEL = "claude-fable-5"

# Hard cap on Reason -> Act loops before LangGraph raises
# GraphRecursionError. Stops a misbehaving tool from looping forever.
RECURSION_LIMIT = 25


# ── 1. STATE ─────────────────────────────────────────────────

class AgentState(TypedDict):
    # operator.add makes every node's returned messages APPEND to the
    # list rather than replace it. This append behaviour is what turns
    # the graph into a running conversation.
    messages: Annotated[list[BaseMessage], operator.add]


# ── 2. TOOLS ─────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports math.cbrt, math.sqrt, math.pow, etc."""
    try:
        # NOTE: eval with an emptied __builtins__ blocks the obvious
        # attacks but is NOT a real sandbox. Fine for this local demo;
        # harden before exposing to untrusted input.
        result = eval(expression, {"__builtins__": {}, "math": math}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_date() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


# Register both tools in a list. Passed to both the LLM and the ToolNode.
tools = [calculator, get_date]


# ── 3. THE LLM ───────────────────────────────────────────────

llm = ChatAnthropic(model=MODEL)
llm_with_tools = llm.bind_tools(tools)


# ── 4. NODES ─────────────────────────────────────────────────

def agent_node(state: AgentState):
    """REASON step — ask Claude what to do next."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)  # ACT + OBSERVE step


# ── 5. ROUTING ───────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # If Claude's response has tool_calls, it wants to Act.
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # No tool_calls means Claude has its final answer.
    return "end"


# ── 6. BUILD THE GRAPH ───────────────────────────────────────

def build_agent():
    """Compile the ReAct graph into a runnable app."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)   # Reason
    graph.add_node("tools", tool_node)    # Act + Observe

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    # After tools run, always go back to agent. This IS the loop.
    graph.add_edge("tools", "agent")

    return graph.compile()


app = build_agent()


# ── 7. TRACE ─────────────────────────────────────────────────

def _message_text(msg: AIMessage) -> str:
    """Extract plain text from an AIMessage.

    With a thinking-enabled model (Fable) the content is a list of
    blocks (thinking + text), not a bare string. Pull out just the
    text blocks and drop thinking blocks.
    """
    content = msg.content
    if isinstance(content, str):
        return content

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def run_events(question: str, recursion_limit: int = RECURSION_LIMIT):
    """Run the agent and yield one normalised event per step.

    Event shapes:
        {"type": "thought",     "tool": <name>, "input": <args>}
        {"type": "observation", "tool": <name>, "result": <str>}
        {"type": "answer",      "content": <str>}

    This walks only the NEW messages produced at each step (msgs[seen:]),
    which is what makes multi-tool turns work: when Claude calls two
    tools at once, ToolNode appends BOTH results in a single step, and
    reading only messages[-1] would silently drop one of them.
    """
    initial_state = {"messages": [HumanMessage(content=question)]}
    seen = 0

    for step in app.stream(
        initial_state,
        stream_mode="values",
        config={"recursion_limit": recursion_limit},
    ):
        messages = step["messages"]

        # Only the messages added since the previous step.
        for msg in messages[seen:]:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # Reason step — Claude decided to use one or more tools.
                    for tc in msg.tool_calls:
                        yield {
                            "type": "thought",
                            "tool": tc["name"],
                            "input": tc["args"],
                        }
                else:
                    # No tool calls — this is the final answer.
                    yield {"type": "answer", "content": _message_text(msg)}

            elif isinstance(msg, ToolMessage):
                # Observe step — a tool returned a result.
                yield {
                    "type": "observation",
                    "tool": msg.name,
                    "result": msg.content,
                }

        seen = len(messages)
