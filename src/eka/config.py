"""Central configuration for the Enterprise Knowledge Assistant.

All paths, model ids, and tunable parameters live here so nodes stay thin.
Values can be overridden via environment variables (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
GOLDEN_SET_PATH = DATA_DIR / "eval" / "golden_set.json"
CHROMA_DIR = PROJECT_ROOT / ".chroma"
MCP_SERVER_SCRIPT = PROJECT_ROOT / "mcp_server" / "server.py"
MCP_DB_PATH = PROJECT_ROOT / "mcp_server" / "service_desk.db"

CHROMA_COLLECTION = "enterprise_kb"

# --- Models ---
# Default to the model the local proxy exposes (ANTHROPIC_MODEL, e.g. via
# Claude Code / tabitoken); fall back to a sane public default otherwise.
_DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
GEN_MODEL = os.getenv("EKA_GEN_MODEL", _DEFAULT_MODEL)
JUDGE_MODEL = os.getenv("EKA_JUDGE_MODEL", _DEFAULT_MODEL)
EMBED_MODEL = os.getenv("EKA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Anthropic auth / endpoint ---
# Two supported modes:
#   1. Direct Anthropic  -> ANTHROPIC_API_KEY (sent as x-api-key).
#   2. Proxy / gateway   -> ANTHROPIC_AUTH_TOKEN (sent as Authorization: Bearer)
#      plus ANTHROPIC_BASE_URL (e.g. tabitoken / OpenRouter-style endpoints).
# When an auth token is present it takes precedence (Bearer auth).
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")

# --- RAG parameters ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
TOP_K_RETRY = 8
RELEVANCE_FLOOR = 0.20  # below this the best hit is treated as "insufficient context"

# --- Evaluation / retry ---
FAITHFULNESS_THRESHOLD = 0.70
ANSWER_RELEVANCY_THRESHOLD = 0.70
MAX_RETRIES = 1

# --- Observability ---
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def require_api_key() -> str:
    """Return the active Anthropic credential or raise a clear error.

    Accepts either a direct API key or a Bearer auth token (proxy mode).
    """
    cred = ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY
    if not cred:
        raise RuntimeError(
            "No Anthropic credential set. Copy .env.example to .env and set "
            "ANTHROPIC_API_KEY (direct) or ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL (proxy)."
        )
    return cred


def make_chat_llm(model: str, **kwargs):
    """Build a ChatAnthropic bound to the configured auth mode + endpoint.

    Centralized so both generation and the RAGAS judge authenticate identically:
    Bearer auth via ANTHROPIC_AUTH_TOKEN when present (proxy/gateway), else a
    direct ANTHROPIC_API_KEY. Imported lazily so config stays import-light.
    """
    from langchain_anthropic import ChatAnthropic

    params: dict = {"model": model, "temperature": 0, **kwargs}
    if ANTHROPIC_BASE_URL:
        params["anthropic_api_url"] = ANTHROPIC_BASE_URL
    if ANTHROPIC_AUTH_TOKEN:  # proxy: Authorization: Bearer <token>
        params["anthropic_api_key"] = ANTHROPIC_AUTH_TOKEN
        params["default_headers"] = {
            **kwargs.get("default_headers", {}),
            "Authorization": f"Bearer {ANTHROPIC_AUTH_TOKEN}",
        }
    else:  # direct Anthropic: x-api-key
        params["anthropic_api_key"] = ANTHROPIC_API_KEY
    return ChatAnthropic(**params)
