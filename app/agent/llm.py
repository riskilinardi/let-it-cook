from google.genai.errors import ServerError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from app.config import MODEL_NAME

# The model can sometimes return a 503 or other transient error. We want to retry those instead of crashing the run.
_TRANSIENT_ERRORS = (ServerError, ChatGoogleGenerativeAIError)


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=temperature)


def resilient(runnable):
    """Retry a model call with exponential backoff on a transient error instead of
    letting it crash the run. Each question makes several calls, so one momentary
    503 should not end the whole thing."""
    return runnable.with_retry(
        retry_if_exception_type=_TRANSIENT_ERRORS,
        wait_exponential_jitter=True,
        stop_after_attempt=5,
    )