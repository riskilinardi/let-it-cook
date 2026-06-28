from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL, RETRIEVER_K


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local embedding model once and reuse it.

    This is the same bge-small-en model from the RAG project. normalize_embeddings
    is on because bge is trained for cosine similarity over unit vectors. Pinned to
    cpu so it runs on any machine. The first call downloads the model (~130MB).
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open the persisted Chroma collection that ingest.py built."""
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


@tool
def search_documents(query: str) -> str:
    """Search the user's own uploaded documents for passages relevant to the query.
    Use this when the question is about the contents of those documents rather than
    general or current world knowledge.
    """
    matches = get_vector_store().similarity_search(query, k=RETRIEVER_K)
    if not matches:
        return "No relevant passages were found in the documents."

    blocks = []
    for doc in matches:
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[source: {source}]\n{doc.page_content}")
    return "\n\n".join(blocks)