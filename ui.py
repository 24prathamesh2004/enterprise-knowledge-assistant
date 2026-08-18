"""Minimal Flask UI for the Enterprise Knowledge Assistant.

Local demo only: binds to 127.0.0.1 with NO authentication. Do not expose this
to a network without adding access controls.

  python ui.py   ->   http://127.0.0.1:5000
"""
from __future__ import annotations

import math

from flask import Flask, jsonify, render_template, request

from graph import build_graph

app = Flask(__name__)
_graph = build_graph()


def _json_safe(obj):
    """Replace non-finite floats (RAGAS emits NaN) with None so it's valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/ask")
def ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    result = _graph.invoke({"question": question})
    return jsonify(
        _json_safe(
            {
                "answer": result.get("answer", ""),
                "evaluation_summary": result.get("evaluation_summary", ""),
                "scores": result.get("ragas_scores", {}),
                "escalated": result.get("escalated", False),
                "ticket_id": result.get("ticket_id"),
                "trace": [e.get("node") for e in result.get("trace", [])],
            }
        )
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
