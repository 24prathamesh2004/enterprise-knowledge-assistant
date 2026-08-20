"""Input validation and context-sufficiency guardrails."""

import re

MAX_QUESTION_CHARS = 2000
RELEVANCE_FLOOR = 0.20

# PII patterns for detection
PII_PATTERNS = {
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
}

# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore.*previous.*instructions",
    r"disregard.*previous",
    r"you are now",
    r"system prompt",
    r"reveal.*prompt",
    r"act as.*dan",
    r"forget.*instructions",
    r"pretend you are",
]


def validate_input(question: str) -> tuple[bool, str]:
    """Validate user input with multiple checks."""
    if not question or not question.strip():
        return False, "Empty question."
    
    if len(question) > MAX_QUESTION_CHARS:
        return False, f"Question too long (max {MAX_QUESTION_CHARS} characters)."
    
    # Check for PII
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, question):
            return False, f"Potential {pii_type.upper()} detected. Please remove sensitive data."
    
    # Check for prompt injection
    injection_pattern = "|".join(INJECTION_PATTERNS)
    if re.search(injection_pattern, question, re.IGNORECASE):
        return False, "Potential prompt injection detected."
    
    return True, ""


def is_context_sufficient(metadata_list: list[dict]) -> bool:
    """Check if retrieved context has sufficient relevance."""
    if not metadata_list:
        return False
    
    best_score = max(m.get("score", 0.0) for m in metadata_list)
    return best_score >= RELEVANCE_FLOOR
