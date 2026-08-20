"""Per-node execution logging for observability."""

import functools
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eka")

INPUT_KEYS = ("question", "top_k", "retry_count", "is_valid")


def preview_value(value, limit: int = 160):
    """Shorten long values for logging."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, list):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return {k: preview_value(v, 80) for k, v in value.items()}
    return value


def log_node(node_name: str):
    """Decorator to log node execution and add to trace."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(state: dict) -> dict:
            inputs = {k: preview_value(state[k]) for k in INPUT_KEYS if k in state}

            start = time.perf_counter()
            result = func(state) or {}
            duration_ms = round((time.perf_counter() - start) * 1000, 1)

            outputs = {k: preview_value(v) for k, v in result.items() if k != "trace"}
            logger.info("node=%s (%sms) in=%s out=%s", node_name, duration_ms, inputs, outputs)

            merged = dict(result)
            merged["trace"] = [{
                "node": node_name,
                "duration_ms": duration_ms,
                "inputs": inputs,
                "outputs": outputs,
            }]
            return merged

        return wrapper
    return decorator


def print_trace(trace: list[dict]):
    """Pretty-print the execution trace."""
    print("\n=== Graph execution trace ===")
    for i, entry in enumerate(trace, 1):
        print(f"{i}. {entry['node']} ({entry['duration_ms']}ms)")
        if entry.get("inputs"):
            print(f"     in:  {entry['inputs']}")
        for key, val in entry.get("outputs", {}).items():
            print(f"     out: {key} = {val}")
