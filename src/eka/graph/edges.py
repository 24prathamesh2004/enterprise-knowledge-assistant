"""Conditional routing functions for the graph."""
from __future__ import annotations

import math

from ..config import ANSWER_RELEVANCY_THRESHOLD, FAITHFULNESS_THRESHOLD, MAX_RETRIES
from ..observability.logging import logger


def route_after_guard(state: dict) -> str:
    """Invalid input skips retrieval/generation and goes straight to finalize."""
    return "retrieve" if state.get("is_valid", True) else "finalize"


def _below(value, threshold: float) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(value) and value < threshold


def decide_after_eval(state: dict) -> str:
    """Retry retrieval once (broader) when RAGAS scores are low; else finalize.

    This is the PDF's bonus conditional edge. NaN scores (eval errors) do not
    trigger a retry, and retry_count caps the loop at MAX_RETRIES.
    """
    scores = state.get("ragas_scores", {})
    retry_count = state.get("retry_count", 0)
    low = _below(scores.get("faithfulness"), FAITHFULNESS_THRESHOLD) or _below(
        scores.get("answer_relevancy"), ANSWER_RELEVANCY_THRESHOLD
    )
    if low and retry_count < MAX_RETRIES:
        logger.info("decide -> retry (low RAGAS scores); broadening retrieval")
        return "retry"
    logger.info("decide -> finalize")
    return "finalize"
