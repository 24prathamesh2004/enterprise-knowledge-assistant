"""RAGAS evaluation for one (question, answer, contexts) sample, plus a
plain-language interpretation of the scores.

`interpret_scores` is a pure function (no heavy imports) so it stays easy to
test. `run_ragas` imports RAGAS lazily and never raises — on any failure it
returns NaN scores so the graph keeps running.
"""
from __future__ import annotations

import math

_METRIC_ORDER = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _label(v: float) -> str:
    if _is_nan(v):
        return "n/a (eval unavailable)"
    if v >= 0.80:
        return "strong"
    if v >= 0.60:
        return "acceptable (minor gaps)"
    return "weak"


def interpret_scores(scores: dict) -> str:
    """Map RAGAS scores to a short, human-readable interpretation."""
    lines: list[str] = []
    for key in _METRIC_ORDER:
        if key in scores:
            val = scores[key]
            shown = "nan" if _is_nan(val) else f"{val:.2f}"
            lines.append(f"- {key}: {shown} — {_label(val)}")

    faith, rel = scores.get("faithfulness"), scores.get("answer_relevancy")
    if _is_nan(faith) or _is_nan(rel):
        summary = "Evaluation incomplete — treat the answer with caution."
    elif faith >= 0.70 and rel >= 0.70:
        summary = "Faithful and on-topic; the answer is well grounded in the sources."
    elif faith < 0.70:
        summary = "Low faithfulness — the answer may contain unsupported claims."
    else:
        summary = "Low answer relevancy — may not directly address the question."
    lines.append(f"Summary: {summary}")
    return "\n".join(lines)


def _load_metrics(with_reference: bool):
    """Import metric classes defensively (names differ across RAGAS versions)."""
    from ragas.metrics import Faithfulness

    try:
        from ragas.metrics import ResponseRelevancy as AnswerRel
    except ImportError:
        from ragas.metrics import AnswerRelevancy as AnswerRel

    metrics = [Faithfulness(), AnswerRel()]
    if with_reference:
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

        metrics += [LLMContextPrecisionWithReference(), LLMContextRecall()]
    return metrics


def _normalize(col: str) -> str | None:
    """Map a RAGAS result column to our canonical metric key (robust to renames)."""
    c = col.lower()
    if "faith" in c:
        return "faithfulness"
    if "relevanc" in c and "context" not in c:
        return "answer_relevancy"
    if "precision" in c:
        return "context_precision"
    if "recall" in c:
        return "context_recall"
    return None


def _ensure_ragas_importable() -> None:
    """Stub the removed `langchain_community.chat_models.vertexai` path.

    RAGAS imports ChatVertexAI at module load, but langchain-community 0.4+
    dropped that submodule. We never use Vertex, so registering a lightweight
    stub lets `import ragas` succeed. Idempotent; no-ops if the real module exists.
    """
    import sys
    import types

    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except Exception:
            stub = types.ModuleType(mod_name)
            stub.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[mod_name] = stub
    try:
        import langchain_community.llms as _llms

        if not hasattr(_llms, "VertexAI"):
            _llms.VertexAI = type("VertexAI", (), {})
    except Exception:
        pass


def run_ragas(question, answer, contexts, reference=None) -> dict:
    """Evaluate one sample with RAGAS. Context precision/recall run only when a
    reference is supplied. Never raises: returns NaN (+ _error) on failure."""
    scores = {"faithfulness": float("nan"), "answer_relevancy": float("nan")}
    try:
        import nest_asyncio

        nest_asyncio.apply()
        _ensure_ragas_importable()
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        try:
            from ragas import EvaluationDataset, SingleTurnSample
        except ImportError:
            from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

        from config import JUDGE_MODEL, make_llm
        from rag import get_embeddings

        # bypass_n: Groq's chat API rejects n>1, but answer_relevancy asks for
        # 3 question generations. This fans them out into 3 separate n=1 calls.
        judge = LangchainLLMWrapper(make_llm(JUDGE_MODEL, max_tokens=4096), bypass_n=True)
        embeddings = LangchainEmbeddingsWrapper(get_embeddings())
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=list(contexts),
            reference=reference,
        )
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=_load_metrics(with_reference=bool(reference)),
            llm=judge,
            embeddings=embeddings,
        )
        df = result.to_pandas()
        for col in df.columns:
            key = _normalize(col)
            if key is not None:
                try:
                    scores[key] = float(df[col].mean())
                except (TypeError, ValueError):
                    continue
        return scores
    except Exception as exc:  # eval must never break the graph
        scores["_error"] = str(exc)
        return scores
