"""Chunk documents with a heading-aware recursive splitter."""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import CHUNK_OVERLAP, CHUNK_SIZE

# Split on headings first so a policy clause stays intact, then paragraphs/lines.
_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " "]
_HEADING_RE = re.compile(r"^#{1,3}\s+(.*)$", re.MULTILINE)


def _first_heading(text: str) -> str | None:
    m = _HEADING_RE.search(text)
    return m.group(1).strip() if m else None


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks tagged with source/section/chunk_id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )
    chunks = splitter.split_documents(docs)
    for i, ch in enumerate(chunks):
        ch.metadata["chunk_id"] = i
        section = _first_heading(ch.page_content)
        if section:
            ch.metadata["section"] = section
    return chunks
