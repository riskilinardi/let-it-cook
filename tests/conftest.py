import pytest
from langchain_core.tools import tool

from app.agent.graph import build_turn_input, make_agent


class FakeChatModel:
    """Returns scripted responses instead of calling an API, so the graph can run
    offline. bind_tools is a no-op because the scripts already carry any tool calls,
    and with_structured_output hands back the scripted Critique objects."""

    def __init__(self, responses, critiques=None):
        self._responses = list(responses)
        self._critiques = list(critiques or [])

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return _FakeStructured(self._critiques)

    def invoke(self, messages, config=None):
        assert self._responses, "fake model ran out of scripted responses"
        return self._responses.pop(0)


class _FakeStructured:
    def __init__(self, critiques):
        self._critiques = critiques

    def invoke(self, prompt, config=None):
        assert self._critiques, "fake model ran out of scripted critiques"
        return self._critiques.pop(0)


@tool
def web_search(query: str) -> str:
    """fake web search"""
    return "fake web result"


@tool
def search_documents(query: str) -> str:
    """fake document search"""
    return "fake document result"


@tool
def calculator(expression: str) -> str:
    """fake calculator"""
    return "fake calc result"


FAKE_TOOLS = [web_search, search_documents, calculator]


@pytest.fixture
def run_fake():
    def _run(responses, critiques):
        agent = make_agent(llm=FakeChatModel(responses, critiques), tools=FAKE_TOOLS)
        return agent.invoke(build_turn_input("test"), {"configurable": {"thread_id": "test"}})
    return _run