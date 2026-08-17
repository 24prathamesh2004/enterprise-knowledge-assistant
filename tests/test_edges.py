"""Routing / conditional-edge unit tests (pure)."""
from eka.graph.edges import decide_after_eval, route_after_guard


def test_route_valid_goes_to_retrieve():
    assert route_after_guard({"is_valid": True}) == "retrieve"


def test_route_invalid_goes_to_finalize():
    assert route_after_guard({"is_valid": False}) == "finalize"


def test_decide_retries_on_low_faithfulness():
    state = {"ragas_scores": {"faithfulness": 0.5, "answer_relevancy": 0.9}, "retry_count": 0}
    assert decide_after_eval(state) == "retry"


def test_decide_retries_on_low_relevancy():
    state = {"ragas_scores": {"faithfulness": 0.9, "answer_relevancy": 0.4}, "retry_count": 0}
    assert decide_after_eval(state) == "retry"


def test_decide_finalizes_when_scores_high():
    state = {"ragas_scores": {"faithfulness": 0.9, "answer_relevancy": 0.9}, "retry_count": 0}
    assert decide_after_eval(state) == "finalize"


def test_decide_finalizes_when_retry_budget_exhausted():
    state = {"ragas_scores": {"faithfulness": 0.1, "answer_relevancy": 0.1}, "retry_count": 1}
    assert decide_after_eval(state) == "finalize"


def test_decide_does_not_retry_on_nan():
    state = {"ragas_scores": {"faithfulness": float("nan"), "answer_relevancy": float("nan")}, "retry_count": 0}
    assert decide_after_eval(state) == "finalize"
