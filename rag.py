"""RAG pipeline: load Markdown policies, chunk them, embed locally, persist to
Chroma, and retrieve the most relevant chunks for a question.

`build_index()` is the one-time offline step; `retrieve()` is the runtime step
the graph's Retriever node calls on every query.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    KNOWLEDGE_BASE_DIR,
    TOP_K,
)

# Split on headings first so a policy clause stays intact, then paragraphs/lines.
_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " "]
_HEADING_RE = re.compile(r"^#{1,3}\s+(.*)$", re.MULTILINE)


def load_documents() -> list[Document]:
    """Read every Markdown file in the knowledge base into a Document."""
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KNOWLEDGE_BASE_DIR}")
    docs = [
        Document(page_content=path.read_text(encoding="utf-8"),
                 metadata={"source": path.name})
        for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))
    ]
    if not docs:
        raise FileNotFoundError(f"No .md files found in {KNOWLEDGE_BASE_DIR}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks tagged with source/section/chunk_id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=_SEPARATORS,
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        heading = _HEADING_RE.search(chunk.page_content)
        if heading:
            chunk.metadata["section"] = heading.group(1).strip()
    return chunks


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Local sentence-transformer embeddings, shared by the store and RAGAS.

    token=False forces anonymous HF Hub access so a stale/expired token in the
    environment can't turn this public model's download into a 401.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"token": False},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index() -> int:
    """Embed the knowledge base into a persistent Chroma collection.

    Returns the number of chunks stored. Run once before querying.
    """
    chunks = chunk_documents(load_documents())
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )
    return len(chunks)


@lru_cache(maxsize=1)
def _store() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(question: str, top_k: int = TOP_K) -> tuple[list[str], list[dict]]:
    """Return (chunk texts, metadata) for the top_k most similar chunks.

    Metadata carries source, section, and a normalized relevance score in [0, 1].
    """
    hits = _store().similarity_search_with_relevance_scores(question, k=top_k)
    contexts = [doc.page_content for doc, _ in hits]
    meta = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "section": doc.metadata.get("section"),
            "score": float(score),
        }
        for doc, score in hits
    ]
    return contexts, meta
