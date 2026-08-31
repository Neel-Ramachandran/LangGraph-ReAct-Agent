# ============================================================
#  react_agent.py
#  Runs the ReAct agent from the command line and prints the
#  full trace (thoughts, tool calls, observations, final answer).
#
#  The agent itself lives in agent.py; this file only handles
#  presentation.
#
#  Run:
#    python react_agent.py
# ============================================================

from langgraph.errors import GraphRecursionError

from agent import run_events, RECURSION_LIMIT


def run_agent(question: str):
    print(f"\n{'='*55}")
    print(f"  Question: {question}")
    print(f"{'='*55}")

    try:
        for event in run_events(question):
            if event["type"] == "thought":
                print(f"\n  [THOUGHT]  I need to use: {event['tool']}")
                print(f"             Input: {event['input']}")
            elif event["type"] == "observation":
                print(f"\n  [OBSERVE]  {event['tool']} returned: {event['result']}")
            elif event["type"] == "answer":
                print(f"\n  [ANSWER]   {event['content']}")
    except GraphRecursionError:
        print(
            f"\n  [STOPPED]  Hit the {RECURSION_LIMIT}-step limit without "
            f"finishing — the agent may be stuck in a loop."
        )

    print(f"\n{'='*55}\n")


# ── EXAMPLES ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Uses both tools in a single turn.
    run_agent("What day will it be 30 days from now, and what is 150 percent of 100?")
