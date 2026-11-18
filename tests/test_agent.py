import pytest
from google.genai.errors import ClientError, ServerError
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from app.agent.graph import Critique
from tests.conftest import FAKE_TOOLS


def _server_error(code):
    return ServerError(code, {"error": {"code": code, "message": "overloaded", "status": "UNAVAILABLE"}})


def _client_error(code):
    # how langchain surfaces a 4xx: its own error type, with the SDK error as the cause
    wrapped = ChatGoogleGenerativeAIError(f"Error calling model ({code})")
    wrapped.__cause__ = ClientError(code, {"error": {"code": code, "message": "client error", "status": "X"}})
    return wrapped


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


def test_a_transient_model_error_is_retried(run_fake):
    responses = [
        AIMessage(content="plan"),
        _server_error(503),  # the first reasoning call fails once
        AIMessage(content="The answer."),
    ]
    state = run_fake(responses, [Critique(sufficient=True, feedback="fine")])
    assert state["answer"] == "The answer."


def test_a_rate_limited_call_is_retried_and_so_is_the_reflector(run_fake):
    responses = [_client_error(429), AIMessage(content="plan"), AIMessage(content="The answer.")]
    critiques = [_server_error(500), Critique(sufficient=True, feedback="fine")]
    state = run_fake(responses, critiques)
    assert state["answer"] == "The answer."


def test_a_permanent_model_error_is_not_retried(run_fake):
    # a bad API key fails the same way every time, so it should surface at once.
    # if it were retried, the fake would hand back the plan instead of raising.
    responses = [_client_error(400), AIMessage(content="plan"), AIMessage(content="never reached")]
    with pytest.raises(ChatGoogleGenerativeAIError):
        run_fake(responses, [Critique(sufficient=True, feedback="fine")])


def _tool_results(state):
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def test_an_unknown_tool_becomes_an_error_result(run_fake):
    responses = [
        AIMessage(content="plan"),
        _tool_call("stock_prices", {"ticker": "GOOG"}),
        AIMessage(content="I could not look that up."),
    ]
    state = run_fake(responses, [Critique(sufficient=True, feedback="fine")])
    [result] = _tool_results(state)
    assert result.status == "error"
    assert "no tool called 'stock_prices'" in result.content
    assert state["answer"] == "I could not look that up."


@tool
def broken_search(query: str) -> str:
    """a search tool whose backend is down"""
    raise ConnectionError("search backend unreachable")


def test_a_failing_tool_becomes_an_error_result(run_fake):
    responses = [
        AIMessage(content="plan"),
        _tool_call("broken_search", {"query": "anything"}),
        AIMessage(content="Search was down, so here is what I know."),
    ]
    state = run_fake(responses, [Critique(sufficient=True, feedback="fine")], tools=FAKE_TOOLS + [broken_search])
    [result] = _tool_results(state)
    assert result.status == "error"
    assert "broken_search failed (ConnectionError)" in result.content
    assert state["answer"].startswith("Search was down")