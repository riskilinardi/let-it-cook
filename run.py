import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import run_agent


def current_turn_messages(messages: list) -> list:
    """Return only the messages from the latest user question onward, so the trace
    shows this turn rather than the whole conversation history."""
    start = 0
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage) and message.name != "reflection":
            start = i
    return messages[start:]


def show_trace(state) -> None:
    """Print this turn: the plan, each tool call and result, the reflection
    verdicts, and the final answer."""
    print("\nPLAN")
    print(state["plan"])

    print("\nSTEPS")
    for message in current_turn_messages(state["messages"]):
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"  tool: {call['name']}  input: {call['args']}")
        elif isinstance(message, ToolMessage):
            preview = str(message.content)[:300].replace("\n", " ")
            print(f"  result: {preview}...")

    if state["reflections"]:
        print("\nREFLECTION")
        for note in state["reflections"]:
            print(f"  {note}")

    print("\nANSWER")
    print(state["answer"])


def main() -> None:
    # one question on the command line runs a single turn
    if len(sys.argv) > 1:
        show_trace(run_agent(" ".join(sys.argv[1:])))
        return

    # otherwise start a conversation. the shared thread_id is what gives the agent
    # memory of earlier turns. type quit to leave.
    print("Ask the agent questions. Type 'quit' to exit.\n")
    thread_id = "session"
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if question:
            show_trace(run_agent(question, thread_id=thread_id))
            print()


if __name__ == "__main__":
    main()