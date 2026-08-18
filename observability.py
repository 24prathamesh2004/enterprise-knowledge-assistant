"""Manual per-node observability.

`log_node` wraps a graph node so each execution records the node name, a preview
of its inputs and outputs, and its duration — both to stdout and into
`state["trace"]` (an append-reduced list). This guarantees the node-by-node
execution log the assignment requires, with or without LangSmith.
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eka")

# State keys worth surfacing as a node's "inputs" in the trace.
_INPUT_KEYS = ("question", "top_k", "retry_count", "is_valid")


def _preview(value: Any, limit: int = 160) -> Any:
    """Compact a value for logging without dumping huge payloads."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, list):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return {k: _preview(v, 80) for k, v in value.items()}
    return value


def log_node(name: str) -> Callable:
    """Decorator: log a node's inputs, outputs, and duration; append a trace entry."""

    def decorator(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            inputs = {k: _preview(state[k]) for k in _INPUT_KEYS if k in state}
            start = time.perf_counter()
            result = fn(state) or {}
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            outputs = {k: _preview(v) for k, v in result.items() if k != "trace"}
            logger.info("node=%s (%sms) in=%s out=%s", name, duration_ms, inputs, outputs)
            merged = dict(result)
            merged["trace"] = [
                {"node": name, "duration_ms": duration_ms, "inputs": inputs, "outputs": outputs}
            ]
            return merged

        return wrapper

    return decorator


def print_trace(trace: list[dict]) -> None:
    """Pretty-print the accumulated node trace (used by the CLI)."""
    print("\n=== Graph execution trace ===")
    for i, entry in enumerate(trace, 1):
        print(f"{i}. {entry['node']} ({entry['duration_ms']}ms)")
        if entry.get("inputs"):
            print(f"     in:  {entry['inputs']}")
        for key, val in entry.get("outputs", {}).items():
            print(f"     out: {key} = {val}")
