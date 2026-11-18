import json
import logging
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app.api.main as main
import app.api.streaming as streaming
from app.agent.graph import Critique, make_agent
from app.api.ratelimit import DAILY_MESSAGE, RateLimited, RateLimiter
from app.config import DEFAULT_ALLOWED_ORIGINS, parse_allowed_origins
from tests.conftest import FAKE_TOOLS, FakeChatModel

QUESTION = {"question": "What is 2 + 2?", "thread_id": "test-thread"}


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    # every test starts with the default limits and nothing counted yet
    monkeypatch.setattr(main, "limiter", RateLimiter(per_minute=5, per_day=100))


@pytest.fixture
def fake_run(monkeypatch):
    """Replace the agent behind /ask with a stub, and record what it was called with."""
    calls = []

    def run_agent(question, thread_id):
        calls.append({"question": question, "thread_id": thread_id})
        return {"answer": "4", "plan": "add", "reflections": ["approved"]}

    monkeypatch.setattr(main, "run_agent", run_agent)
    return calls


def _client(ip="203.0.113.10"):
    return TestClient(main.app, client=(ip, 50000))


def _events(body: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]


def test_health():
    response = _client().get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("question", ["", "x" * 2001])
def test_empty_or_too_long_questions_are_rejected(question, fake_run):
    response = _client().post("/ask", json={"question": question})
    assert response.status_code == 422
    assert fake_run == []


@pytest.mark.parametrize("thread_id", ["", "x" * 65, "../etc", "has spaces"])
def test_bad_thread_ids_are_rejected(thread_id, fake_run):
    response = _client().post("/ask", json={"question": "hi", "thread_id": thread_id})
    assert response.status_code == 422


def test_a_missing_thread_id_gets_a_fresh_one(fake_run):
    client = _client()
    client.post("/ask", json={"question": "first"})
    client.post("/ask", json={"question": "second"})
    first, second = (call["thread_id"] for call in fake_run)
    assert first != second
    assert first != "default"


def test_ask_returns_the_answer(fake_run):
    response = _client().post("/ask", json=QUESTION)
    assert response.status_code == 200
    assert response.json()["answer"] == "4"
    assert fake_run == [{"question": QUESTION["question"], "thread_id": "test-thread"}]


def test_the_per_ip_limit_returns_429(fake_run):
    client = _client()
    for _ in range(5):
        assert client.post("/ask", json=QUESTION).status_code == 200

    response = client.post("/ask/stream", json=QUESTION)  # counted across both endpoints
    assert response.status_code == 429
    assert response.json()["limit"] == "per_ip"
    assert 1 <= int(response.headers["Retry-After"]) <= 60

    # someone else is not affected
    assert _client(ip="198.51.100.7").post("/ask", json=QUESTION).status_code == 200


def test_the_daily_cap_returns_429_with_its_own_message(monkeypatch, fake_run):
    monkeypatch.setattr(main, "limiter", RateLimiter(per_minute=100, per_day=2))
    assert _client("192.0.2.1").post("/ask", json=QUESTION).status_code == 200
    assert _client("192.0.2.2").post("/ask", json=QUESTION).status_code == 200

    response = _client("192.0.2.3").post("/ask", json=QUESTION)
    assert response.status_code == 429
    assert response.json() == {"detail": DAILY_MESSAGE, "limit": "daily"}
    assert int(response.headers["Retry-After"]) <= 24 * 60 * 60


def test_rejected_requests_do_not_use_up_the_allowance(monkeypatch, fake_run):
    monkeypatch.setattr(main, "limiter", RateLimiter(per_minute=1, per_day=1))
    client = _client()
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json=QUESTION).status_code == 200


def test_the_window_slides_and_the_day_resets_at_midnight_utc():
    now = [datetime(2026, 11, 19, 23, 59, 0, tzinfo=timezone.utc).timestamp()]
    limiter = RateLimiter(per_minute=1, per_day=2, clock=lambda: now[0])

    limiter.check("a")
    with pytest.raises(RateLimited) as exc:
        limiter.check("a")
    assert exc.value.limit == "per_ip"

    now[0] += 30
    limiter.check("b")  # the second question of the day
    with pytest.raises(RateLimited) as exc:
        limiter.check("c")
    assert exc.value.limit == "daily"
    assert exc.value.retry_after == 30  # 23:59:30 -> midnight

    now[0] += 31  # 00:00:01 the next day: the count starts again
    limiter.check("c")


def test_allowed_origins_parsing():
    assert parse_allowed_origins(None) == DEFAULT_ALLOWED_ORIGINS
    assert parse_allowed_origins("") == DEFAULT_ALLOWED_ORIGINS
    assert parse_allowed_origins(" , ") == DEFAULT_ALLOWED_ORIGINS
    assert parse_allowed_origins(
        " https://letitcook-rl.vercel.app/ ,http://localhost:3000"
    ) == ["https://letitcook-rl.vercel.app", "http://localhost:3000"]


def test_cors_allows_only_the_configured_origins():
    def preflight(origin):
        return _client().options("/ask/stream", headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })

    allowed = preflight("http://localhost:3000")
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in preflight("https://elsewhere.example").headers


def test_the_stream_sends_each_step_then_done(monkeypatch):
    agent = make_agent(
        llm=FakeChatModel(
            [AIMessage(content="plan"), AIMessage(content="It is 4.")],
            [Critique(sufficient=True, feedback="fine")],
        ),
        tools=FAKE_TOOLS,
    )
    monkeypatch.setattr(streaming, "get_agent", lambda: agent)

    response = _client().post("/ask/stream", json=QUESTION)
    types = [e["type"] for e in _events(response.text)]
    assert types == ["plan", "reflection", "answer", "done"]


def test_the_stream_sends_an_error_event_when_the_agent_fails(monkeypatch, caplog):
    secret_detail = "internal detail that must not reach the browser"
    agent = make_agent(llm=FakeChatModel([RuntimeError(secret_detail)]), tools=FAKE_TOOLS)
    monkeypatch.setattr(streaming, "get_agent", lambda: agent)

    with caplog.at_level(logging.ERROR, logger="app"):
        response = _client().post("/ask/stream", json=QUESTION)

    assert response.status_code == 200
    events = _events(response.text)
    assert [e["type"] for e in events] == ["error", "done"]
    assert events[0]["message"] == streaming.STREAM_ERROR_MESSAGE
    assert secret_detail not in response.text
    assert secret_detail in caplog.text  # but it is in the server log, with the trace


def test_the_log_records_the_question_length_but_not_its_text(caplog, fake_run):
    question = "a very particular question about pelicans"
    with caplog.at_level(logging.INFO, logger="app"):
        _client().post("/ask", json={"question": question})

    assert f"question accepted: {len(question)} chars" in caplog.text
    assert "POST /ask 200" in caplog.text
    assert "pelicans" not in caplog.text


def test_tracing_is_off_during_tests():
    from langsmith.utils import tracing_is_enabled
    assert not tracing_is_enabled()
