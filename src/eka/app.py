"""CLI entry point: run one question through the assistant graph."""
from __future__ import annotations

import argparse

from .graph import build_graph
from .observability.logging import print_trace


def run_once(question: str, reference: str | None = None) -> dict:
    """Build the graph and run a single question through it."""
    graph = build_graph()
    state: dict = {"question": question}
    if reference:
        state["reference"] = reference
    return graph.invoke(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Knowledge Assistant")
    parser.add_argument("--question", "-q", required=True, help="Question to ask")
    parser.add_argument("--reference", help="Optional ground truth (enables context metrics)")
    args = parser.parse_args()

    print("=== Enterprise Knowledge Assistant ===")
    graph = build_graph()
    try:
        print(graph.get_graph().draw_ascii())
    except Exception:  # grandalf not installed — fall back to mermaid text
        print(graph.get_graph().draw_mermaid())

    state: dict = {"question": args.question}
    if args.reference:
        state["reference"] = args.reference
    result = graph.invoke(state)

    print_trace(result.get("trace", []))
    print("\n=== Answer ===")
    print(result.get("answer", ""))
    print("\n=== RAGAS evaluation ===")
    print(result.get("evaluation_summary", "(no evaluation)"))
    print("\n=== MCP service desk ===")
    print(f"audit: {result.get('mcp_log')}")
    if result.get("escalated"):
        print(f"escalated -> ticket {result.get('ticket_id')}")


if __name__ == "__main__":
    main()
