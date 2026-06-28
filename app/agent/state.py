from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Original qns
    question: str
    # Planner's output
    plan: str
    # Working scratchpad. Every result will land here.
    messages: Annotated[list[AnyMessage], add_messages]
    # How many loops have been run so far.
    iterations: int
    # How many times the model reflected and revised its plan.
    revisions: int
    # Whether the current draft passed the reflection check.
    approved: bool
    # Short notes from each reflection pass.
    reflections: list[str]
    # Final answer, filled in when the agent is done.
    answer: str