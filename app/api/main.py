import hashlib
import logging
import os
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import run_agent
from app.api.ratelimit import RateLimited, RateLimiter
from app.api.streaming import stream_agent
from app.config import ALLOWED_ORIGINS, DAILY_QUESTION_CAP, RATE_LIMIT_PER_MINUTE

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    # uvicorn sets up only its own loggers, so give the app's modules (app.*) a
    # handler of their own. Third-party loggers are left alone and stay quiet.
    app_logger = logging.getLogger("app")
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        app_logger.addHandler(handler)
        app_logger.setLevel(logging.INFO)


_configure_logging()

# a fresh salt on every start, so the client tags in the log can't be turned back
# into IP addresses, but stay stable for as long as the server runs
_CLIENT_SALT = os.urandom(16)


def _client_tag(scope) -> str:
    """A short salted hash of the client IP. Enough to tell visitors apart in the
    log (and to check that each one shows up as their own IP behind the proxy)
    without writing addresses down."""
    client = scope.get("client")
    host = client[0] if client else "unknown"
    return hashlib.sha256(_CLIENT_SALT + host.encode()).hexdigest()[:8]


class RequestLogMiddleware:
    """Log one line per request: method, path, status, latency and a client tag.
    For a stream, the latency covers the whole run. An unhandled error is logged
    with its stack trace and answered with a plain 500."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status = None

        async def send_and_record(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_and_record)
        except Exception:
            logger.exception("unhandled error on %s %s", scope["method"], scope["path"])
            if status is not None:
                raise  # the response already started, so there is nothing more to send
            status = 500
            await JSONResponse({"detail": "Internal server error"}, status_code=500)(scope, receive, send)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "%s %s %s %.0fms client=%s",
                scope["method"], scope["path"], status, elapsed_ms, _client_tag(scope),
            )


app = FastAPI(title="Research Agent API")

# middleware added later wraps the earlier ones, so CORS sits outside the request
# log and its headers also go on the log's 500 responses
app.add_middleware(RequestLogMiddleware)

# the frontend runs on a different origin, and the browser blocks cross-origin
# calls unless that origin is allowed here. set ALLOWED_ORIGINS on the host to the
# deployed frontend's URL; see app/config.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = RateLimiter(RATE_LIMIT_PER_MINUTE, DAILY_QUESTION_CAP)

logger.info(
    "allowed origins: %s | limits: %d questions/min per IP, %d/day in total",
    ", ".join(ALLOWED_ORIGINS), RATE_LIMIT_PER_MINUTE, DAILY_QUESTION_CAP,
)


@app.exception_handler(RateLimited)
async def rate_limited(request: Request, exc: RateLimited) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": exc.message, "limit": exc.limit},
        headers={"Retry-After": str(exc.retry_after)},
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # a request without a thread_id gets a fresh one, so separate visitors never
    # share conversation memory. the frontend sends a UUID per conversation.
    thread_id: str = Field(
        default_factory=lambda: str(uuid4()),
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


def _admit(request: Request, question: str) -> None:
    """Apply the rate limits to a question that passed validation, so a malformed
    request never uses up anyone's allowance. Logs the question's length, never
    its text."""
    limiter.check(request.client.host if request.client else "unknown")
    logger.info("question accepted: %d chars", len(question))


@app.get("/")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(body: AskRequest, request: Request) -> dict:
    """Run the agent and return the finished result in one response. Easy to test,
    but the caller waits for the whole run with nothing to show in the meantime."""
    _admit(request, body.question)
    state = run_agent(body.question, thread_id=body.thread_id)
    return {
        "answer": state["answer"],
        "plan": state["plan"],
        "reflections": state["reflections"],
    }


@app.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Run the agent and stream each step as a Server-Sent Event as it happens. This
    is the endpoint the frontend consumes to show the reasoning live."""
    _admit(request, body.question)
    return StreamingResponse(
        stream_agent(body.question, body.thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
