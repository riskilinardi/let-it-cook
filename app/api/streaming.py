import json
import logging

from app.agent.graph import build_run_config, build_turn_input, get_agent

logger = logging.getLogger(__name__)

# What the visitor sees when a run fails. The details stay in the server log.
STREAM_ERROR_MESSAGE = "Something went wrong while the agent was working on this. Please try again."


def _events_from_update(node: str, update: dict) -> list[dict]:
    """Turn one node's state update into zero or more events for the client.

    A reason step that called tools becomes one tool_call event per call. An act
    step becomes one observation event per result. plan, reflect and finalize each
    become a single event. A reason step that only wrote a draft (no tool calls)
    produces nothing here; the reflection that follows it carries the visible signal.
    """
    if node == "plan":
        return [{"type": "plan", "plan": update["plan"]}]

    if node == "reason":
        message = update["messages"][0]
        return [
            {"type": "tool_call", "tool": call["name"], "input": call["args"]}
            for call in message.tool_calls
        ]

    if node == "act":
        events = []
        for message in update["messages"]:
            text = str(message.content)
            preview = text[:600] + ("..." if len(text) > 600 else "")
            events.append({"type": "observation", "tool": message.name, "result": preview})
        return events

    if node == "reflect":
        return [{
            "type": "reflection",
            "note": update["reflections"][-1],
            "approved": update["approved"],
        }]

    if node == "finalize":
        return [{"type": "answer", "answer": update["answer"]}]

    return []


def _sse(event: dict) -> str:
    """Format a dict as one Server-Sent Event: a data: line closed by a blank line."""
    return f"data: {json.dumps(event)}\n\n"


async def stream_agent(question: str, thread_id: str):
    """Run the agent and yield its steps as SSE events as they happen.

    stream_mode="updates" makes LangGraph hand back each node's output the moment
    that node finishes, which is what lets us surface the plan, tool calls,
    observations and reflections live rather than only at the end.
    """
    config = build_run_config(question, thread_id)
    turn_input = build_turn_input(question)

    # the response has already started by the time anything here fails, so an
    # exception can't become an HTTP error status. Send it as an event instead,
    # so the client can show it in the turn rather than the stream just stopping.
    try:
        async for chunk in get_agent().astream(turn_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                for event in _events_from_update(node, update):
                    yield _sse(event)
    except Exception:
        logger.exception("agent run failed while streaming")
        yield _sse({"type": "error", "message": STREAM_ERROR_MESSAGE})

    yield _sse({"type": "done"})