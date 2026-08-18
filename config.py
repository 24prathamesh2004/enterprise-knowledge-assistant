"""Central configuration for the Enterprise Knowledge Assistant.

Every tunable — model ids, paths, chunking, thresholds — lives here so the rest
of the code stays declarative. Any value can be overridden via an environment
variable (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"
GOLDEN_SET_PATH = ROOT / "golden_set.json"
CHROMA_DIR = ROOT / ".chroma"
MCP_SERVER_SCRIPT = ROOT / "mcp_server.py"
CHROMA_COLLECTION = "enterprise_kb"

# --- Models: Groq for generation + evaluation; embeddings run locally ---
# Groq deprecated the llama-3.x models in mid-2026; the GPT-OSS family is the
# current production line. A fast model answers; a stronger one judges.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEN_MODEL = os.getenv("EKA_GEN_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("EKA_JUDGE_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = os.getenv("EKA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # Groq has no embeddings API

# --- RAG parameters ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
TOP_K_RETRY = 8            # broadened retrieval on the bonus retry
RELEVANCE_FLOOR = 0.20     # below this, the best hit is treated as "insufficient context"

# --- RAGAS thresholds / retry ---
FAITHFULNESS_THRESHOLD = 0.70
ANSWER_RELEVANCY_THRESHOLD = 0.70
MAX_RETRIES = 1


def make_llm(model: str, **kwargs):
    """Build a ChatGroq client for `model`. Imported lazily so importing config
    stays cheap and free of heavy dependencies."""
    from langchain_groq import ChatGroq

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=0, **kwargs)
