"""Structured per-node logging.

`log_node` wraps a LangGraph node so every execution records the node name,
a preview of its outputs and its duration both to stdout and into
``state["trace"]`` (which uses an append reducer). This guarantees a
node-by-node execution log even without LangSmith.
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
    """Decorator: log a node's outputs + duration and append a trace entry."""

    def decorator(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            start = time.perf_counter()
            result = fn(state) or {}
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            outputs = {k: _preview(v) for k, v in result.items() if k != "trace"}
            logger.info("node=%s (%sms) -> %s", name, duration_ms, outputs)
            entry = {"node": name, "duration_ms": duration_ms, "outputs": outputs}
            merged = dict(result)
            merged["trace"] = [entry]  # append reducer merges into full trace
            return merged

        return wrapper

    return decorator


def print_trace(trace: list[dict]) -> None:
    """Pretty-print the accumulated node trace (used by the CLI)."""
    print("\n=== Graph execution trace ===")
    for i, entry in enumerate(trace, 1):
        print(f"{i}. {entry['node']} ({entry['duration_ms']}ms)")
        for key, val in entry.get("outputs", {}).items():
            print(f"     {key}: {val}")
