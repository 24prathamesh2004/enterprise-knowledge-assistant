"""Assemble the LangGraph StateGraph.

Flow: guard_input → retrieve → generate → evaluate → [decide]
      → {prepare_retry → retrieve | finalize} → END
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .edges import decide_after_eval, route_after_guard
from .nodes import (
    evaluate,
    finalize,
    generate,
    guard_input,
    prepare_retry,
    retrieve,
)
from .state import GraphState


def build_graph():
    """Build and compile the assistant graph."""
    g = StateGraph(GraphState)
    g.add_node("guard_input", guard_input)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("evaluate", evaluate)
    g.add_node("prepare_retry", prepare_retry)
    g.add_node("finalize", finalize)

    g.set_entry_point("guard_input")
    g.add_conditional_edges(
        "guard_input", route_after_guard,
        {"retrieve": "retrieve", "finalize": "finalize"},
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "evaluate")
    g.add_conditional_edges(
        "evaluate", decide_after_eval,
        {"retry": "prepare_retry", "finalize": "finalize"},
    )
    g.add_edge("prepare_retry", "retrieve")
    g.add_edge("finalize", END)
    return g.compile()
