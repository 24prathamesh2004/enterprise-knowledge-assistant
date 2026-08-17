"""Shared LangGraph state passed between all nodes."""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class GraphState(TypedDict, total=False):
    question: str                  # INPUT
    reference: Optional[str]        # golden-set ground truth; enables bonus metrics
    is_valid: bool                 # input guard verdict
    guardrail_reason: Optional[str]
    top_k: int                     # retrieval breadth (bumped on retry)
    retrieved_contexts: list       # chunk texts (RAGAS "contexts")
    retrieved_meta: list           # {source, section, score} per chunk
    answer: str                    # generated answer
    ragas_scores: dict             # {faithfulness, answer_relevancy, ...}
    evaluation_summary: str        # human-readable interpretation
    retry_count: int               # bounds the conditional retry
    escalated: bool                # MCP ticket filed?
    ticket_id: Optional[str]
    mcp_log: Optional[dict]        # MCP audit result
    error: Optional[str]
    trace: Annotated[list, operator.add]  # append reducer: node-by-node log
