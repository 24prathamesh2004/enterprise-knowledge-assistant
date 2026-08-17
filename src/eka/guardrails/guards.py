"""Lightweight guardrails.

- ``validate_input``: rejects empty/oversized input and obvious prompt-injection
  or clearly out-of-scope requests before any model call.
- ``is_context_sufficient``: decides whether retrieval was strong enough to
  answer, otherwise the generator should refuse instead of hallucinating.
"""
from __future__ import annotations

import re

from ..config import RELEVANCE_FLOOR

MAX_QUESTION_CHARS = 2000

# Obvious prompt-injection / jailbreak markers (case-insensitive).
_INJECTION_PATTERNS = [
    r"ignore (all |the )?previous instructions",
    r"disregard (all |the )?(previous|above)",
    r"you are now",
    r"system prompt",
    r"reveal your (system )?prompt",
    r"act as (an?|the) (dan|jailbreak)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def validate_input(question: str) -> tuple[bool, str | None]:
    """Return (is_valid, reason). reason is None when valid."""
    if question is None or not question.strip():
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
    best = max((m.get("score", 0.0) or 0.0) for m in meta)
    return best >= floor
