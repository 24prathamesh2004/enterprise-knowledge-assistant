"""Chroma vector store: build (offline) and load (runtime)."""
from __future__ import annotations

import os
from functools import lru_cache

# The embedding model (BAAI/bge-small-en-v1.5) is public. Don't implicitly attach
# any stored HF token — an expired/foreign token turns an anonymous-OK download
# into a 401. An explicitly exported HF_TOKEN still wins via setdefault.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import CHROMA_COLLECTION, CHROMA_DIR, EMBED_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Local sentence-transformer embeddings (shared by store + RAGAS).

    ``token=False`` forces anonymous HF Hub access for this public model, so a
    stale/expired stored token in the environment can't turn the download into a
    401 (import-order-independent, unlike the HF_HUB_DISABLE_IMPLICIT_TOKEN env).
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"token": False},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Embed chunks and persist them to the Chroma directory."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )


@lru_cache(maxsize=1)
def load_vectorstore() -> Chroma:
    """Open the persisted Chroma collection for querying."""
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
