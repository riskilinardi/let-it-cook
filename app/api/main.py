from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import run_agent
from app.api.streaming import stream_agent

app = FastAPI(title="Research Agent API")

# the Next.js dev server runs on a different origin (localhost:3000), and the
# browser blocks cross-origin calls unless we allow that origin here. origins must
# be exact: scheme, host and port, with no trailing slash. add the deployed
# frontend's URL to this list when you ship in Phase 8.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str = "default"


@app.get("/")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    """Run the agent and return the finished result in one response. Easy to test,
    but the caller waits for the whole run with nothing to show in the meantime."""
    state = run_agent(request.question, thread_id=request.thread_id)
    return {
        "answer": state["answer"],
        "plan": state["plan"],
        "reflections": state["reflections"],
    }


@app.post("/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    """Run the agent and stream each step as a Server-Sent Event as it happens. This
    is the endpoint the frontend consumes to show the reasoning live."""
    return StreamingResponse(
        stream_agent(request.question, request.thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )