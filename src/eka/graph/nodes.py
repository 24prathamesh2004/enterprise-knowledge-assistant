"""LangGraph node functions: guard → retrieve → generate → evaluate → finalize."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import (
    ANSWER_RELEVANCY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    GEN_MODEL,
    TOP_K,
)
from ..evaluation.ragas_eval import interpret_scores, run_ragas
from ..guardrails.guards import is_context_sufficient, validate_input
from ..mcp.client import call_service_desk
from ..observability.logging import log_node
from ..rag.retriever import retrieve as retrieve_chunks

_REFUSAL = (
    "I don't have enough information in the knowledge base to answer that "
    "confidently. This has been logged for a human to follow up."
)

_SYSTEM_PROMPT = (
    "You are Northwind Technologies' internal knowledge assistant. "
    "Answer ONLY using the provided context. If the context does not contain "
    "the answer, say you don't know — never invent policy details. "
    "Cite the source file name(s) you used in parentheses."
)


def _format_context(contexts, meta) -> str:
    blocks = []
    for i, text in enumerate(contexts):
        src = meta[i].get("source", "unknown") if i < len(meta) else "unknown"
        blocks.append(f"[Source: {src}]\n{text}")
    return "\n\n---\n\n".join(blocks)


@log_node("guard_input")
def guard_input(state: dict) -> dict:
    """Validate the question and initialize retrieval params."""
    valid, reason = validate_input(state.get("question", ""))
    updates = {
        "is_valid": valid,
        "guardrail_reason": reason,
        "top_k": TOP_K,
        "retry_count": 0,
    }
    if not valid:
        updates["answer"] = f"Request not processed: {reason}"
    return updates


@log_node("retrieve")
def retrieve(state: dict) -> dict:
    """Similarity-search the vector store for the current top_k."""
    try:
        contexts, meta = retrieve_chunks(state["question"], state.get("top_k", TOP_K))
        return {"retrieved_contexts": contexts, "retrieved_meta": meta}
    except Exception as exc:  # keep the graph alive
        return {"retrieved_contexts": [], "retrieved_meta": [], "error": str(exc)}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _invoke_llm(messages):
    from ..config import make_chat_llm

    llm = make_chat_llm(GEN_MODEL, max_tokens=1024)
    return llm.invoke(messages)


@log_node("generate")
def generate(state: dict) -> dict:
    """Generate a grounded answer, or refuse when context is insufficient."""
    contexts = state.get("retrieved_contexts", [])
    meta = state.get("retrieved_meta", [])
    if not is_context_sufficient(meta):
        return {"answer": _REFUSAL}
    context_block = _format_context(contexts, meta)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context_block}\n\nQuestion: {state['question']}"
        ),
    ]
    try:
        resp = _invoke_llm(messages)
        return {"answer": resp.content}
    except Exception as exc:  # keep the graph alive
        return {"answer": _REFUSAL, "error": f"generation failed: {exc}"}


@log_node("evaluate")
def evaluate(state: dict) -> dict:
    """Score the answer with RAGAS and produce an interpretation."""
    scores = run_ragas(
        question=state["question"],
        answer=state.get("answer", ""),
        contexts=state.get("retrieved_contexts", []),
        reference=state.get("reference"),
    )
    return {"ragas_scores": scores, "evaluation_summary": interpret_scores(scores)}


def _is_low_confidence(scores: dict) -> bool:
    """True when a present metric is below threshold (NaN alone does not trigger)."""
    import math

    faith = scores.get("faithfulness")
    rel = scores.get("answer_relevancy")
    low = False
    if isinstance(faith, (int, float)) and not math.isnan(faith):
        low = low or faith < FAITHFULNESS_THRESHOLD
    if isinstance(rel, (int, float)) and not math.isnan(rel):
        low = low or rel < ANSWER_RELEVANCY_THRESHOLD
    return low


@log_node("finalize")
def finalize(state: dict) -> dict:
    """Audit the interaction via MCP; escalate genuine knowledge gaps to a ticket."""
    scores = state.get("ragas_scores", {})
    meta = state.get("retrieved_meta", [])
    is_valid = state.get("is_valid", True)
    # Escalate only real questions we could not answer well — not junk/injection.
    escalate = is_valid and (
        not is_context_sufficient(meta) or _is_low_confidence(scores)
    )
    summary = f"Unanswered/low-confidence query: {state.get('question', '')[:200]}"
    try:
        return call_service_desk(
            question=state.get("question", ""),
            answer=state.get("answer", ""),
            scores=scores,
            escalate=escalate,
            escalate_summary=summary,
        )
    except Exception as exc:  # MCP failure must not lose the answer
        return {"escalated": False, "ticket_id": None, "mcp_log": {"error": str(exc)}}


@log_node("prepare_retry")
def prepare_retry(state: dict) -> dict:
    """Broaden retrieval for one more attempt after a low RAGAS score."""
    from ..config import TOP_K_RETRY

    return {"retry_count": state.get("retry_count", 0) + 1, "top_k": TOP_K_RETRY}
