"""The LangGraph assistant: shared state, the three agent nodes (Retriever,
Response, Evaluator), the guard/finalize nodes, the bonus retry edge, and graph
assembly.

Flow: guard_input -> retrieve -> generate -> evaluate -> [decide]
      -> { prepare_retry -> retrieve | finalize } -> END
"""
from __future__ import annotations

import math
import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from config import (
    ANSWER_RELEVANCY_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    GEN_MODEL,
    MAX_RETRIES,
    TOP_K,
    TOP_K_RETRY,
    make_llm,
)
from guards import is_context_sufficient, validate_input
from mcp_client import call_service_desk
from observability import log_node
from rag import retrieve as retrieve_chunks
from ragas_eval import interpret_scores, run_ragas


class GraphState(TypedDict, total=False):
    question: str
    reference: Optional[str]        # golden-set ground truth; enables bonus metrics
    is_valid: bool
    guardrail_reason: Optional[str]
    top_k: int
    retrieved_contexts: list
    retrieved_meta: list
    answer: str
    ragas_scores: dict
    evaluation_summary: str
    retry_count: int
    escalated: bool
    ticket_id: Optional[str]
    mcp_log: Optional[dict]
    error: Optional[str]
    trace: Annotated[list, operator.add]  # append reducer: node-by-node log


_SYSTEM_PROMPT = (
    "You are Northwind Technologies' internal knowledge assistant. "
    "Answer ONLY using the provided context. If the context does not contain "
    "the answer, say you don't know — never invent policy details. "
    "Keep distinct categories separate — do not add or combine figures (e.g. "
    "annual leave and sick leave) into a total unless the question asks for one. "
    "Cite the source file name(s) you used in parentheses."
)
_REFUSAL = (
    "I don't have enough information in the knowledge base to answer that "
    "confidently. This has been logged for a human to follow up."
)


def _format_context(contexts: list[str], meta: list[dict]) -> str:
    blocks = []
    for i, text in enumerate(contexts):
        source = meta[i].get("source", "unknown") if i < len(meta) else "unknown"
        blocks.append(f"[Source: {source}]\n{text}")
    return "\n\n---\n\n".join(blocks)


@log_node("guard_input")
def guard_input(state: dict) -> dict:
    """Validate the question and initialize retrieval params (innovation)."""
    is_valid, reason = validate_input(state.get("question", ""))
    updates = {"is_valid": is_valid, "guardrail_reason": reason, "top_k": TOP_K, "retry_count": 0}
    if not is_valid:
        updates["answer"] = f"Request not processed: {reason}"
    return updates


@log_node("retrieve")
def retrieve(state: dict) -> dict:
    """Retriever agent: similarity-search the vector store for the current top_k."""
    try:
        contexts, meta = retrieve_chunks(state["question"], state.get("top_k", TOP_K))
    except Exception as exc:  # index missing/unreadable — refuse instead of crashing
        return {"retrieved_contexts": [], "retrieved_meta": [], "error": f"retrieval failed: {exc}"}
    return {"retrieved_contexts": contexts, "retrieved_meta": meta}


@log_node("generate")
def generate(state: dict) -> dict:
    """Response agent: answer strictly from retrieved context, or refuse."""
    contexts = state.get("retrieved_contexts", [])
    meta = state.get("retrieved_meta", [])
    if not is_context_sufficient(meta):
        return {"answer": _REFUSAL}
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{_format_context(contexts, meta)}\n\nQuestion: {state['question']}"
        ),
    ]
    try:
        answer = make_llm(GEN_MODEL, max_tokens=1024).invoke(messages).content
    except Exception as exc:  # Groq/network error — degrade to a safe refusal
        return {"answer": _REFUSAL, "error": f"generation failed: {exc}"}
    return {"answer": answer}


@log_node("evaluate")
def evaluate(state: dict) -> dict:
    """Evaluator agent: score the answer with RAGAS and interpret it."""
    scores = run_ragas(
        question=state["question"],
        answer=state.get("answer", ""),
        contexts=state.get("retrieved_contexts", []),
        reference=state.get("reference"),
    )
    return {"ragas_scores": scores, "evaluation_summary": interpret_scores(scores)}


@log_node("prepare_retry")
def prepare_retry(state: dict) -> dict:
    """Broaden retrieval for one more attempt after a low RAGAS score."""
    return {"retry_count": state.get("retry_count", 0) + 1, "top_k": TOP_K_RETRY}


@log_node("finalize")
def finalize(state: dict) -> dict:
    """MCP consumer: audit every interaction; escalate genuine knowledge gaps."""
    scores = state.get("ragas_scores", {})
    # Escalate only real questions we answered poorly — not junk/injection.
    escalate = state.get("is_valid", True) and (
        not is_context_sufficient(state.get("retrieved_meta", [])) or _is_low_confidence(scores)
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


def _below(value, threshold: float) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(value) and value < threshold


def _is_low_confidence(scores: dict) -> bool:
    """True when a present metric is below threshold (NaN alone does not trigger)."""
    return _below(scores.get("faithfulness"), FAITHFULNESS_THRESHOLD) or _below(
        scores.get("answer_relevancy"), ANSWER_RELEVANCY_THRESHOLD
    )


def route_after_guard(state: dict) -> str:
    """Invalid input skips retrieval/generation and goes straight to finalize."""
    return "retrieve" if state.get("is_valid", True) else "finalize"


def decide_after_eval(state: dict) -> str:
    """Bonus conditional edge: retry retrieval once when RAGAS scores are low.

    NaN scores (eval errors) do not trigger a retry; retry_count caps the loop.
    """
    if _is_low_confidence(state.get("ragas_scores", {})) and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "finalize"


def build_graph():
    """Assemble and compile the assistant graph."""
    g = StateGraph(GraphState)
    g.add_node("guard_input", guard_input)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("evaluate", evaluate)
    g.add_node("prepare_retry", prepare_retry)
    g.add_node("finalize", finalize)

    g.set_entry_point("guard_input")
    g.add_conditional_edges(
        "guard_input", route_after_guard, {"retrieve": "retrieve", "finalize": "finalize"}
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "evaluate")
    g.add_conditional_edges(
        "evaluate", decide_after_eval, {"retry": "prepare_retry", "finalize": "finalize"}
    )
    g.add_edge("prepare_retry", "retrieve")
    g.add_edge("finalize", END)
    return g.compile()
