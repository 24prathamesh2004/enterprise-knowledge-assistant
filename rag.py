"""RAG pipeline using LangChain components."""

import os
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    KNOWLEDGE_BASE_DIR,
    TOP_K,
)

SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " "]
HEADING_PATTERN = re.compile(r"^#{1,3}\s+(.*)$", re.MULTILINE)

embeddings_instance = None
vector_store_instance = None


class LocalEmbeddings(Embeddings):
    """Sentence-transformers embeddings loaded locally, without HuggingFace auth."""

    def __init__(self, model_name: str):
        # An expired HF token in the environment breaks the local model load.
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGINGFACE_TOKEN", None)
        self.model = SentenceTransformer(model_name, use_auth_token=False)
        self.model_name = model_name  # RAGAS reads this attribute directly

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


def get_embeddings() -> Embeddings:
    """Get singleton embeddings instance using local-only approach."""
    global embeddings_instance
    if embeddings_instance is None:
        embeddings_instance = LocalEmbeddings(EMBED_MODEL)
    return embeddings_instance


def get_vector_store() -> Chroma:
    """Get singleton vector store instance."""
    global vector_store_instance
    if vector_store_instance is None:
        vector_store_instance = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    return vector_store_instance


def load_documents() -> list[Document]:
    """Load every policy document from the knowledge base."""
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()
    if not docs:
        raise FileNotFoundError(f"No .txt files in {KNOWLEDGE_BASE_DIR}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents using LangChain text splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        heading = HEADING_PATTERN.search(chunk.page_content)
        if heading:
            chunk.metadata["section"] = heading.group(1).strip()

    return chunks


def build_index() -> int:
    """Build vector index using LangChain."""
    global vector_store_instance
    
    docs = load_documents()
    chunks = chunk_documents(docs)
    
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    vector_store_instance = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )
    
    return len(chunks)


def retrieve_with_scores(question: str, top_k: int = TOP_K) -> tuple[list[str], list[dict]]:
    """Retrieve documents with real similarity scores from Chroma."""
    store = get_vector_store()
    # similarity_search_with_score returns (Document, distance) pairs.
    # Chroma uses L2 distance by default: distance=0 is perfect, higher is worse.
    # Convert to a 0-1 similarity score: score = 1 / (1 + distance)
    results = store.similarity_search_with_score(question, k=top_k)

    contexts = [doc.page_content for doc, _ in results]
    metadata = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "section": doc.metadata.get("section"),
            "score": round(1.0 / (1.0 + float(distance)), 4),
        }
        for doc, distance in results
    ]

    return contexts, metadata
