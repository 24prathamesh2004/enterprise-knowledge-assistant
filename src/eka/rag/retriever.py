"""Similarity retrieval over the Chroma store."""
from __future__ import annotations

from ..config import TOP_K
from .vectorstore import load_vectorstore


def retrieve(question: str, top_k: int = TOP_K) -> tuple[list[str], list[dict]]:
    """Return (context texts, metadata) for the top_k most similar chunks.

    Metadata carries source, section and a normalized relevance score in [0, 1].
    """
    vs = load_vectorstore()
    hits = vs.similarity_search_with_relevance_scores(question, k=top_k)
    contexts: list[str] = []
    meta: list[dict] = []
    for doc, score in hits:
        contexts.append(doc.page_content)
        meta.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "section": doc.metadata.get("section"),
                "score": float(score),
            }
        )
    return contexts, meta
