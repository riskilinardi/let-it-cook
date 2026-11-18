from google.genai.errors import ClientError, ServerError
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from app.agent.llm import _model_kwargs, is_transient


def _wrapped(code):
    error = ChatGoogleGenerativeAIError("Error calling model")
    error.__cause__ = ClientError(code, {"error": {"code": code, "message": "x", "status": "X"}})
    return error


def test_retries_only_rate_limits_and_server_errors():
    assert is_transient(ServerError(503, {"error": {"code": 503, "message": "x", "status": "X"}}))
    assert is_transient(ServerError(500, {"error": {"code": 500, "message": "x", "status": "X"}}))
    assert is_transient(_wrapped(429))
    assert not is_transient(_wrapped(400))  # malformed request
    assert not is_transient(_wrapped(403))  # bad or revoked API key
    assert not is_transient(ChatGoogleGenerativeAIError("unsupported message type"))  # no status at all
    assert not is_transient(ValueError("something else"))


def test_temperature_is_left_unset_on_gemini_3():
    assert "temperature" not in _model_kwargs("gemini-3-flash", 0.0)
    assert _model_kwargs("gemini-2.5-flash-lite", 0.0)["temperature"] == 0.0


def test_sdk_retries_are_off_and_a_timeout_is_set():
    kwargs = _model_kwargs("gemini-2.5-flash-lite", 0.0)
    assert kwargs["max_retries"] == 1
    assert kwargs["timeout"] > 0
