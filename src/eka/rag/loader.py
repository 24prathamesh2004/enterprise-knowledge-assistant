"""Load knowledge-base documents from disk into LangChain Documents."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from ..config import KB_DIR


def load_documents(kb_dir: Path | None = None) -> list[Document]:
    """Read every Markdown file in the knowledge base into a Document."""
    kb_dir = Path(kb_dir) if kb_dir else KB_DIR
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")
    docs: list[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    if not docs:
        raise FileNotFoundError(f"No .md documents found in {kb_dir}")
    return docs
