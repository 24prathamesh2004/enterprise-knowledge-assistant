"""Loader + splitter tests (light deps: langchain-core + text-splitters)."""
from eka.rag.loader import load_documents
from eka.rag.splitter import chunk_documents


def test_loader_reads_all_markdown_docs():
    docs = load_documents()
    assert len(docs) >= 10
    assert all(d.page_content.strip() for d in docs)
    assert all(d.metadata.get("source", "").endswith(".md") for d in docs)


def test_chunking_produces_more_chunks_than_docs():
    docs = load_documents()
    chunks = chunk_documents(docs)
    assert len(chunks) >= len(docs)
    assert all("chunk_id" in c.metadata for c in chunks)
    assert all("source" in c.metadata for c in chunks)


def test_chunk_size_respected():
    docs = load_documents()
    chunks = chunk_documents(docs)
    # allow a small margin over CHUNK_SIZE for separator boundaries
    assert all(len(c.page_content) <= 1200 for c in chunks)
