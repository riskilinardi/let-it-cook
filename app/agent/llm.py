from google.genai.errors import APIError
from langchain_core.runnables.retry import RunnableRetry
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry_if_exception

from app.config import MODEL_NAME, MODEL_TIMEOUT_SECONDS

# Backoff between retries: tenacity's wait_exponential_jitter parameters, in seconds.
RETRY_WAIT = {"initial": 1, "max": 10}
RETRY_ATTEMPTS = 5


def _model_kwargs(model_name: str, temperature: float) -> dict:
    kwargs = {
        "model": model_name,
        "timeout": MODEL_TIMEOUT_SECONDS,
        # The Google SDK retries on its own by default (6 attempts). Turn that off
        # so resilient() below is the only retry layer; 1 means a single attempt.
        "max_retries": 1,
    }
    # Gemini 3 models misbehave at low temperatures, so leave theirs unset and let
    # the model use its default.
    if not model_name.startswith("gemini-3"):
        kwargs["temperature"] = temperature
    return kwargs


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(**_model_kwargs(MODEL_NAME, temperature))


def _status_code(exc: BaseException) -> int | None:
    """The HTTP status behind a model error, if there is one. A 5xx arrives as the
    SDK's ServerError. A 4xx (429 included) is re-raised by langchain as
    ChatGoogleGenerativeAIError with the SDK's ClientError as its __cause__."""
    for err in (exc, exc.__cause__):
        if isinstance(err, APIError):
            return err.code
    return None


def is_transient(exc: BaseException) -> bool:
    """Worth retrying: rate limited (429) or a server-side failure (5xx). Anything
    else, like a bad API key or a malformed request, fails the same way every time."""
    code = _status_code(exc)
    return code is not None and (code == 429 or 500 <= code < 600)


class _RetryTransient(RunnableRetry):
    """RunnableRetry can only filter by exception type, but one type
    (ChatGoogleGenerativeAIError) covers both a 429 and a bad API key. Filter by
    status code instead."""

    @property
    def _kwargs_retrying(self) -> dict:
        kwargs = super()._kwargs_retrying
        kwargs["retry"] = retry_if_exception(is_transient)
        return kwargs


def resilient(runnable):
    """Retry a model call with exponential backoff on a transient error instead of
    letting it crash the run. Each question makes several calls, so one momentary
    503 should not end the whole thing."""
    return _RetryTransient(
        bound=runnable,
        kwargs={},
        config={},
        wait_exponential_jitter=True,
        exponential_jitter_params=dict(RETRY_WAIT),
        max_attempt_number=RETRY_ATTEMPTS,
    )
