import shutil
from pathlib import Path

import pypdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agent.retriever import get_embeddings
from app.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
)


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = pypdf.PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def load_documents() -> list[Document]:
    documents = []
    for path in sorted(Path(DOCUMENTS_DIR).glob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        text = _read(path)
        if text.strip():
            documents.append(Document(page_content=text, metadata={"source": path.name}))
        else:
            print(f"Skipped '{path.name}': no extractable text.")
    return documents


def main() -> None:
    documents = load_documents()
    if not documents:
        print(f"No readable .txt, .md or .pdf files in '{DOCUMENTS_DIR}/'. Add some and rerun.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Remove any existing vector store to avoid mixing old and new data
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    print(f"Loaded {len(documents)} document(s), split into {len(chunks)} chunks.")
    print("Embedding and writing the vector store (a large PDF takes a minute)...")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done. Vector store saved to '{CHROMA_DIR}/'.")


if __name__ == "__main__":
    main()