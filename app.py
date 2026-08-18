"""Single entry point for the Enterprise Knowledge Assistant.

  python app.py --build-index          # embed the knowledge base into Chroma
  python app.py --ask "your question"  # run one question through the graph
  python app.py --eval                 # run the golden set + aggregate RAGAS
"""
from __future__ import annotations

import argparse
import json

from config import GOLDEN_SET_PATH

_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def build_index() -> None:
    from rag import build_index as _build

    print(f"Indexed {_build()} chunks into the vector store.")


def ask(question: str) -> None:
    from graph import build_graph
    from observability import print_trace

    graph = build_graph()
    try:
        print(graph.get_graph().draw_ascii())
    except Exception:  # grandalf not installed — fall back to mermaid text
        print(graph.get_graph().draw_mermaid())

    result = graph.invoke({"question": question})
    print_trace(result.get("trace", []))
    print("\n=== Answer ===\n" + result.get("answer", ""))
    print("\n=== RAGAS evaluation ===\n" + result.get("evaluation_summary", "(no evaluation)"))
    print(f"\n=== MCP service desk ===\naudit: {result.get('mcp_log')}")
    if result.get("escalated"):
        print(f"escalated -> ticket {result.get('ticket_id')}")


def evaluate_golden_set() -> None:
    from graph import build_graph

    items = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    graph = build_graph()
    collected = {m: [] for m in _METRICS}
    for item in items:
        scores = graph.invoke(
            {"question": item["question"], "reference": item.get("reference")}
        ).get("ragas_scores", {})
        for m in _METRICS:
            value = scores.get(m)
            if isinstance(value, (int, float)) and value == value:  # skip NaN
                collected[m].append(value)
        print(f"Q: {item['question']}\n   {scores}")

    print("\n=== Aggregate (mean over golden set) ===")
    for m in _METRICS:
        values = collected[m]
        mean = sum(values) / len(values) if values else float("nan")
        print(f"{m:22s}: {mean:.3f}  (n={len(values)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise Knowledge Assistant")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-index", action="store_true", help="Embed the knowledge base")
    action.add_argument("--ask", metavar="QUESTION", help="Ask one question")
    action.add_argument("--eval", action="store_true", help="Run the golden set")
    args = parser.parse_args()

    if args.build_index:
        build_index()
    elif args.eval:
        evaluate_golden_set()
    else:
        ask(args.ask)


if __name__ == "__main__":
    main()
