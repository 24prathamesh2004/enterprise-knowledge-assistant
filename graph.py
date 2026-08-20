import math
import operator
from typing import Annotated, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
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
from rag import retrieve_with_scores
from ragas_eval import interpret_scores, run_ragas


class GraphState(TypedDict, total=False):
    question: str
    reference: Optional[str]
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
    chat_history: list
    session_id: Optional[str]
    trace: Annotated[list, operator.add]



PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are Northwind Technologies' internal knowledge assistant.
Answer ONLY using the provided context. If the context does not contain the answer,
say you don't know — never invent details. Keep distinct categories separate.
Cite the source file name(s) in parentheses.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
])

REFUSAL_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that "
    "confidently. This has been logged for human follow-up."
)

# Pre-built RAG chain components (created once, reused)
llm_instance = None
rag_chain_cache = {}


def format_contexts(contexts: list[str], metadata: list[dict]) -> str:
    """Format pre-retrieved context strings for the prompt."""
    parts = []
    for ctx, meta in zip(contexts, metadata):
        source = meta.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{ctx}")
    return "\n\n---\n\n".join(parts)


def get_rag_chain():
    """Get or create the generation chain (prompt → LLM → parser). No retriever inside."""
    global llm_instance

    if llm_instance is None:
        llm_instance = make_llm(GEN_MODEL, max_tokens=1024)

    if "generation" not in rag_chain_cache:
        rag_chain_cache["generation"] = (
            RunnableParallel({
                "context": lambda x: x["context"],
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", []),
            })
            | PROMPT_TEMPLATE
            | llm_instance
            | StrOutputParser()
        )

    return rag_chain_cache["generation"]


# Session-based memory store
memory_store = {}


def get_memory(session_id: str = "default"):
    """Get or create memory for specific session."""
    if session_id not in memory_store:
        memory_store[session_id] = []
    return memory_store[session_id]


@log_node("guard_input")
def guard_input(state: dict) -> dict:
    """Validate input using guardrails."""
    is_valid, reason = validate_input(state.get("question", ""))
    updates = {
        "is_valid": is_valid,
        "guardrail_reason": reason,
        "top_k": TOP_K,
        "retry_count": 0,
        "session_id": state.get("session_id", "default"),
    }
    
    if not is_valid:
        updates["answer"] = f"Request not processed: {reason}"
    
    return updates


@log_node("retrieve")
def retrieve(state: dict) -> dict:
    """Retrieve relevant context using LangChain retriever."""
    contexts, metadata = retrieve_with_scores(
        state["question"], 
        state.get("top_k", TOP_K)
    )
    return {
        "retrieved_contexts": contexts,
        "retrieved_meta": metadata,
    }


@log_node("generate")
def generate(state: dict) -> dict:
    """Generate answer using pre-retrieved contexts from state via LCEL chain."""
    metadata = state.get("retrieved_meta", [])

    if not is_context_sufficient(metadata):
        return {"answer": REFUSAL_MESSAGE}

    context_str = format_contexts(state.get("retrieved_contexts", []), metadata)
    chain = get_rag_chain()

    session_id = state.get("session_id", "default")
    chat_history = get_memory(session_id)

    answer = chain.invoke({
        "input": state["question"],
        "context": context_str,
        "chat_history": chat_history,
    })

    chat_history.append(HumanMessage(content=state["question"]))
    chat_history.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "chat_history": chat_history,
    }


@log_node("evaluate")
def evaluate(state: dict) -> dict:
    """Evaluate answer with RAGAS."""
    scores = run_ragas(
        question=state["question"],
        answer=state.get("answer", ""),
        contexts=state.get("retrieved_contexts", []),
        reference=state.get("reference"),
    )
    
    return {
        "ragas_scores": scores,
        "evaluation_summary": interpret_scores(scores),
    }


@log_node("prepare_retry")
def prepare_retry(state: dict) -> dict:
    """Broaden retrieval for retry."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "top_k": TOP_K_RETRY,
    }


@log_node("finalize")
def finalize(state: dict) -> dict:
    """Audit via MCP and escalate if needed."""
    scores = state.get("ragas_scores", {})
    
    should_escalate = state.get("is_valid", True) and (
        not is_context_sufficient(state.get("retrieved_meta", []))
        or is_low_confidence(scores)
    )
    
    summary = f"Unanswered/low-confidence query: {state.get('question', '')[:200]}"
    
    return call_service_desk(
        question=state.get("question", ""),
        answer=state.get("answer", ""),
        scores=scores,
        escalate=should_escalate,
        escalate_summary=summary,
    )


def is_below_threshold(value, threshold: float) -> bool:
    """Check if value is below threshold."""
    return isinstance(value, (int, float)) and not math.isnan(value) and value < threshold


def is_low_confidence(scores: dict) -> bool:
    """Check if RAGAS scores are low."""
    return (
        is_below_threshold(scores.get("faithfulness"), FAITHFULNESS_THRESHOLD)
        or is_below_threshold(scores.get("answer_relevancy"), ANSWER_RELEVANCY_THRESHOLD)
    )


def route_after_guard(state: dict) -> str:
    """Route based on input validation."""
    return "retrieve" if state.get("is_valid", True) else "finalize"


def decide_after_eval(state: dict) -> str:
    """Conditional retry logic."""
    if is_low_confidence(state.get("ragas_scores", {})) and state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "finalize"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("guard_input", guard_input)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)
    graph.add_node("prepare_retry", prepare_retry)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("guard_input")

    graph.add_conditional_edges(
        "guard_input",
        route_after_guard,
        {"retrieve": "retrieve", "finalize": "finalize"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        decide_after_eval,
        {"retry": "prepare_retry", "finalize": "finalize"},
    )
    graph.add_edge("prepare_retry", "retrieve")
    graph.add_edge("finalize", END)

    return graph.compile()
