import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Disable expired HuggingFace token
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGINGFACE_TOKEN", None)

# Paths
ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = ROOT / "knowledge_base"
GOLDEN_SET_PATH = ROOT / "golden_set.json"
CHROMA_DIR = ROOT / ".chroma"
MCP_SERVER_SCRIPT = ROOT / "mcp_server.py"
CHROMA_COLLECTION = "enterprise_kb"

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Models
GEN_MODEL = os.getenv("EKA_GEN_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("EKA_JUDGE_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = os.getenv("EKA_EMBED_MODEL", "all-MiniLM-L6-v2")

# RAG parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
TOP_K_RETRY = 8
RELEVANCE_FLOOR = 0.20

# RAGAS thresholds
FAITHFULNESS_THRESHOLD = 0.70
ANSWER_RELEVANCY_THRESHOLD = 0.70
MAX_RETRIES = 1


def make_llm(model: str, **kwargs):
    from langchain_groq import ChatGroq
    
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    
    return ChatGroq(
        model=model,
        api_key=GROQ_API_KEY,
        temperature=0,
        **kwargs
    )
