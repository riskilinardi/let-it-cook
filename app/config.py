from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = "gemini-2.5-flash-lite"
SEARCH_MAX_RESULTS = 4
MAX_ITERATIONS = 5
MAX_REVISIONS = 2

# Retriever configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_store"
CHROMA_COLLECTION = "documents"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 4