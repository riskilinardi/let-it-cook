from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agent.llm import get_llm
from app.agent.state import AgentState
from app.config import MAX_ITERATIONS, MAX_REVISIONS


PLANNER_PROMPT = """You are the planning step of a research assistant.

Given the user's question, write a short plan for answering it: the few concrete
steps you would take. The assistant has three tools: a web search for current or
general facts, a document search over the user's own uploaded documents, and a
calculator for arithmetic. Plan around which tool each step needs.

Do not answer the question here. Only lay out the approach, in three or four
short lines. Write them as plain text. Do not use markdown, bold, asterisks, or
numbered headings."""


REASONER_PROMPT = """You are a research assistant answering the user's question.
Today's date is {today}.

You have three tools:
- web_search: current events or general world knowledge.
- search_documents: anything about the contents of the user's own uploaded
  documents. Prefer this when the question is about those documents.
- calculator: any arithmetic. Use it instead of doing the sums yourself.

Work through the plan one step at a time, calling one tool at a time with a
focused input. When you have enough to answer accurately, stop and write the
final answer as plain text. Only state what your tool results support, and name
the sources you used. If a reviewer has left feedback on an earlier draft, treat
fixing it as your priority. If the question needs no tools, just answer it.

The plan you made earlier:
{plan}"""


REFLECTION_PROMPT = """You are a strict reviewer checking a draft answer before it
reaches the user.

Question:
{question}

Draft answer:
{draft}

Judge the draft on three things: does it answer every part of the question, are
the facts specific rather than vague, and does it avoid guessing or hedging where
a real answer was expected. If it falls short on any of these it is not
sufficient, and your feedback should say concretely what to fix or what still
needs to be looked up, in one or two plain sentences with no markdown or bold.
If it answers the question well, mark it sufficient."""


class Critique(BaseModel):
    sufficient: bool = Field(
        description="True only if the draft fully and correctly answers the question."
    )
    feedback: str = Field(
        description="If not sufficient, what is missing or wrong and what to do next."
    )


def make_agent(llm=None, tools=None):
    """Build and compile the agent. Defaults to the real model and tools; tests
    pass fakes so the graph can run offline."""
    if llm is None:
        llm = get_llm()
    if tools is None:
        from app.agent.tools import TOOLS  # imported here so importing this module needs no keys
        tools = TOOLS

    llm_with_tools = llm.bind_tools(tools)
    reflector = llm.with_structured_output(Critique)
    tools_by_name = {t.name: t for t in tools}

    def plan(state: AgentState) -> dict:
        messages = [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=state["question"])]
        return {"plan": llm.invoke(messages).content}

    def reason(state: AgentState) -> dict:
        iterations = state["iterations"] + 1
        system = SystemMessage(
            content=REASONER_PROMPT.format(today=date.today().isoformat(), plan=state["plan"])
        )
        conversation = [system] + state["messages"]
        # at the cap, drop the tools so the model has to answer with what it has
        model = llm if iterations > MAX_ITERATIONS else llm_with_tools
        return {"messages": [model.invoke(conversation)], "iterations": iterations}

    def act(state: AgentState) -> dict:
        results = []
        for call in state["messages"][-1].tool_calls:
            output = tools_by_name[call["name"]].invoke(call["args"])
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    def reflect(state: AgentState) -> dict:
        draft = state["messages"][-1].content
        critique = reflector.invoke(REFLECTION_PROMPT.format(question=state["question"], draft=draft))
        notes = state["reflections"]

        if critique is None or critique.sufficient:
            feedback = critique.feedback if critique else ""
            return {"approved": True, "reflections": notes + [f"approved: {feedback}".strip()]}

        revisions = state["revisions"] + 1
        if revisions > MAX_REVISIONS:
            return {"approved": True, "reflections": notes + [f"accepted, out of revisions: {critique.feedback}"]}

        feedback_message = HumanMessage(
            content=(
                "A reviewer judged your draft answer not good enough. "
                f"{critique.feedback} Address this and write a better answer."
            ),
            name="reflection",
        )
        return {
            "approved": False,
            "revisions": revisions,
            "reflections": notes + [f"needs work: {critique.feedback}"],
            "messages": [feedback_message],
        }

    def finalize(state: AgentState) -> dict:
        return {"answer": state["messages"][-1].content}

    def route_after_reason(state: AgentState) -> str:
        return "act" if state["messages"][-1].tool_calls else "reflect"

    def route_after_reflect(state: AgentState) -> str:
        return "finalize" if state["approved"] else "reason"

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan)
    builder.add_node("reason", reason)
    builder.add_node("act", act)
    builder.add_node("reflect", reflect)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "reason")
    builder.add_conditional_edges("reason", route_after_reason, {"act": "act", "reflect": "reflect"})
    builder.add_edge("act", "reason")
    builder.add_conditional_edges("reflect", route_after_reflect, {"reason": "reason", "finalize": "finalize"})
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=InMemorySaver())


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = make_agent()
    return _agent


def build_run_config(question: str, thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"agent: {question[:50]}",
        "tags": ["research-agent", f"thread:{thread_id}"],
        "metadata": {"thread_id": thread_id, "question": question},
    }


def build_turn_input(question: str) -> dict:
    return {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "plan": "",
        "iterations": 0,
        "revisions": 0,
        "approved": False,
        "reflections": [],
        "answer": "",
    }


def run_agent(question: str, thread_id: str = "default") -> AgentState:
    return get_agent().invoke(build_turn_input(question), config=build_run_config(question, thread_id))