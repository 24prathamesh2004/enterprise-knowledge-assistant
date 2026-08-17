"""Run the golden set through the graph and report aggregate RAGAS scores."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eka.config import GOLDEN_SET_PATH  # noqa: E402
from eka.graph import build_graph  # noqa: E402

_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _mean(vals: list) -> float:
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    items = json.loads(Path(GOLDEN_SET_PATH).read_text(encoding="utf-8"))
    graph = build_graph()
    agg: dict = {m: [] for m in _METRICS}

    for it in items:
        res = graph.invoke(
            {"question": it["question"], "reference": it.get("reference")}
        )
        scores = res.get("ragas_scores", {})
        for m in _METRICS:
            if m in scores:
                agg[m].append(scores[m])
        print(f"Q: {it['question']}")
        print(f"   scores: {scores}")

    print("\n=== Aggregate (mean over golden set) ===")
    for m in _METRICS:
        n = len([v for v in agg[m] if isinstance(v, (int, float)) and v == v])
        print(f"{m:22s}: {_mean(agg[m]):.3f}  (n={n})")


if __name__ == "__main__":
    main()
