import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = "gemini-2.5-flash-lite"
SEARCH_MAX_RESULTS = 4
MAX_ITERATIONS = 5
MAX_REVISIONS = 2

# How long one model request may take before it is abandoned, in seconds.
MODEL_TIMEOUT_SECONDS = 60

# Retriever configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_store"
CHROMA_COLLECTION = "documents"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 4

# Rate limits for the public demo, so it can't quietly run up the API bill. Both
# can be overridden with environment variables on the host.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "5"))  # questions per IP
DAILY_QUESTION_CAP = int(os.getenv("DAILY_QUESTION_CAP", "100"))  # all visitors, resets at midnight UTC

# Frontends allowed to call the API from a browser. Set ALLOWED_ORIGINS to a
# comma-separated list of URLs, such as the deployed Vercel URL.
DEFAULT_ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def parse_allowed_origins(raw: str | None) -> list[str]:
    """Split a comma-separated origin list. The browser compares origins exactly,
    so stray spaces and trailing slashes are stripped. Falls back to the local
    dev server when nothing usable is set."""
    origins = [o.strip().rstrip("/") for o in (raw or "").split(",")]
    origins = [o for o in origins if o]
    return origins or list(DEFAULT_ALLOWED_ORIGINS)


ALLOWED_ORIGINS = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))
