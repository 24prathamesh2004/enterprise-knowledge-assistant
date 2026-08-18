"""Guardrails — the project's innovation layer.

- validate_input: reject empty, oversized, or prompt-injection questions before
  any model call, so junk never reaches retrieval or the LLM.
- is_context_sufficient: when retrieval is too weak, the generator should refuse
  rather than hallucinate an answer.
"""
from __future__ import annotations

import re

from config import RELEVANCE_FLOOR

MAX_QUESTION_CHARS = 2000

_INJECTION_RE = re.compile(
    "|".join(
        [
            r"ignore (all |the )?previous instructions",
            r"disregard (all |the )?(previous|above)",
            r"you are now",
            r"system prompt",
            r"reveal your (system )?prompt",
            r"act as (an?|the) (dan|jailbreak)",
        ]
    ),
    re.IGNORECASE,
)


def validate_input(question: str) -> tuple[bool, str | None]:
    """Return (is_valid, reason); reason is None when the question is accepted."""
    if not question or not question.strip():
        return False, "Empty question."
    if len(question) > MAX_QUESTION_CHARS:
        return False, f"Question exceeds {MAX_QUESTION_CHARS} characters."
    if _INJECTION_RE.search(question):
        return False, "Potential prompt-injection detected."
    return True, None


def is_context_sufficient(meta: list[dict], floor: float = RELEVANCE_FLOOR) -> bool:
    """True when at least one retrieved chunk clears the relevance floor."""
    if not meta:
        return False
    return max((m.get("score", 0.0) or 0.0) for m in meta) >= floor
