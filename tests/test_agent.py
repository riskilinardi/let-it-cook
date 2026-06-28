from langchain_core.messages import AIMessage

from app.agent.graph import Critique


def _tool_call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "1", "type": "tool_call"}])


def test_runs_a_tool_then_answers(run_fake):
    responses = [
        AIMessage(content="step one: search, step two: answer"),
        _tool_call("web_search", {"query": "southeast asia populations"}),
        AIMessage(content="Indonesia is the most populous."),
    ]
    state = run_fake(responses, [Critique(sufficient=True, feedback="clear")])
    assert state["answer"] == "Indonesia is the most populous."
    assert state["plan"].startswith("step one")


def test_reflection_sends_a_weak_answer_back(run_fake):
    responses = [
        AIMessage(content="plan"),
        AIMessage(content="A vague first answer."),
        AIMessage(content="A specific, corrected answer."),
    ]
    critiques = [
        Critique(sufficient=False, feedback="too vague, add the figures"),
        Critique(sufficient=True, feedback="good now"),
    ]
    state = run_fake(responses, critiques)
    assert state["answer"] == "A specific, corrected answer."
    assert state["revisions"] == 1
    assert any("needs work" in note for note in state["reflections"])


def test_revision_cap_stops_the_loop(run_fake):
    responses = [AIMessage(content="plan")] + [AIMessage(content=f"attempt {i}") for i in range(10)]
    critiques = [Critique(sufficient=False, feedback="still not good") for _ in range(10)]
    state = run_fake(responses, critiques)
    assert state["answer"].startswith("attempt")
    assert state["revisions"] <= 2