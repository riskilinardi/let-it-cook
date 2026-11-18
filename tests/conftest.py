import os

# Switch tracing off and blank the API keys before anything imports the app, whose
# config loads .env. load_dotenv never overrides a variable that is already set, so
# tests can't send traces or reach a paid API even with a filled-in local .env.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
for _key in ("GOOGLE_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"):
    os.environ[_key] = ""

import pytest  # noqa: E402
from langchain_core.runnables import Runnable  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

import app.agent.llm as llm_module  # noqa: E402
from app.agent.graph import build_turn_input, make_agent  # noqa: E402


class FakeChatModel(Runnable):
    """Returns scripted responses instead of calling an API, so the graph can run
    offline. bind_tools is a no-op because the scripts already carry any tool calls,
    and with_structured_output hands back the scripted Critique objects. A scripted
    exception is raised instead of returned, to simulate a failing call. It is a
    Runnable so resilient() can wrap it like the real model."""

    def __init__(self, responses, critiques=None):
        self._responses = list(responses)
        self._critiques = list(critiques or [])

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return _FakeStructured(self._critiques)

    def invoke(self, messages, config=None, **kwargs):
        assert self._responses, "fake model ran out of scripted responses"
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeStructured(Runnable):
    def __init__(self, critiques):
        self._critiques = critiques

    def invoke(self, prompt, config=None, **kwargs):
        assert self._critiques, "fake model ran out of scripted critiques"
        critique = self._critiques.pop(0)
        if isinstance(critique, BaseException):
            raise critique
        return critique


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


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    # retries still happen in tests, just without the backoff sleeps
    monkeypatch.setattr(llm_module, "RETRY_WAIT", {"initial": 0, "max": 0, "jitter": 0})


@pytest.fixture
def run_fake():
    def _run(responses, critiques, tools=FAKE_TOOLS):
        agent = make_agent(llm=FakeChatModel(responses, critiques), tools=tools)
        return agent.invoke(build_turn_input("test"), {"configurable": {"thread_id": "test"}})
    return _run
