"""Build the Chroma vector index from the knowledge base (offline step)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eka.rag.loader import load_documents  # noqa: E402
from eka.rag.splitter import chunk_documents  # noqa: E402
from eka.rag.vectorstore import build_vectorstore  # noqa: E402


def main() -> None:
    docs = load_documents()
    print(f"Loaded {len(docs)} documents.")
    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    build_vectorstore(chunks)
    print("Vector store built and persisted to .chroma/")


if __name__ == "__main__":
    main()
