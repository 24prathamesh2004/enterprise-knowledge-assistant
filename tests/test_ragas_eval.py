"""interpret_scores unit tests (pure function, no LLM)."""
from eka.evaluation.ragas_eval import interpret_scores


def test_strong_scores_summary():
    out = interpret_scores({"faithfulness": 0.92, "answer_relevancy": 0.88})
    assert "faithfulness: 0.92" in out
    assert "strong" in out
    assert "well grounded" in out


def test_low_faithfulness_summary():
    out = interpret_scores({"faithfulness": 0.4, "answer_relevancy": 0.9})
    assert "unsupported" in out.lower()


def test_low_relevancy_summary():
    out = interpret_scores({"faithfulness": 0.9, "answer_relevancy": 0.4})
    assert "relevancy" in out.lower()


def test_nan_scores_incomplete():
    out = interpret_scores({"faithfulness": float("nan"), "answer_relevancy": float("nan")})
    assert "incomplete" in out.lower()


def test_context_metrics_included_when_present():
    out = interpret_scores(
        {
            "faithfulness": 0.9,
            "answer_relevancy": 0.9,
            "context_precision": 0.85,
            "context_recall": 0.8,
        }
    )
    assert "context_precision" in out
    assert "context_recall" in out
