"""Main entry point for the Enterprise Knowledge Assistant."""

import argparse
import json
from config import GOLDEN_SET_PATH

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def build_index():
    """Build the vector index from knowledge base."""
    from rag import build_index as build
    num_chunks = build()
    print(f"Indexed {num_chunks} chunks into the vector store.")


def ask_question(question: str):
    """Ask a single question through the graph."""
    from graph import build_graph
    from observability import print_trace
    
    graph = build_graph()

    try:
        print(graph.get_graph().draw_ascii())
    except Exception:
        print(graph.get_graph().draw_mermaid())

    result = graph.invoke({"question": question})

    print_trace(result.get("trace", []))
    print("\n=== Answer ===")
    print(result.get("answer", ""))
    print("\n=== RAGAS evaluation ===")
    print(result.get("evaluation_summary", "(no evaluation)"))
    print("\n=== MCP service desk ===")
    print(f"audit: {result.get('mcp_log')}")

    if result.get("escalated"):
        print(f"escalated -> ticket {result.get('ticket_id')}")


def evaluate_golden_set():
    """Evaluate all questions in the golden set."""
    from graph import build_graph
    
    items = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    graph = build_graph()
    
    collected = {metric: [] for metric in METRICS}
    
    for item in items:
        scores = graph.invoke({
            "question": item["question"],
            "reference": item.get("reference"),
        }).get("ragas_scores", {})
        
        for metric in METRICS:
            value = scores.get(metric)
            if isinstance(value, (int, float)) and value == value:  # skip NaN
                collected[metric].append(value)
        
        print(f"Q: {item['question']}")
        print(f"   {scores}")
    
    print("\n=== Aggregate (mean over golden set) ===")
    for metric in METRICS:
        values = collected[metric]
        mean = sum(values) / len(values) if values else float("nan")
        print(f"{metric:22s}: {mean:.3f}  (n={len(values)})")


def main():
    """Parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(description="Enterprise Knowledge Assistant")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-index", action="store_true", help="Build the vector index")
    action.add_argument("--ask", metavar="QUESTION", help="Ask a question")
    action.add_argument("--eval", action="store_true", help="Evaluate golden set")
    args = parser.parse_args()
    
    if args.build_index:
        build_index()
    elif args.eval:
        evaluate_golden_set()
    else:
        ask_question(args.ask)


if __name__ == "__main__":
    main()
