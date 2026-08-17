"""Run RAGAS metrics on a single (question, answer, contexts) sample and turn
the scores into plain-language interpretation.

RAGAS and the heavy model wrappers are imported lazily inside ``run_ragas`` so
the pure ``interpret_scores`` helper (and its unit tests) stay import-light.
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

    faith = scores.get("faithfulness")
    rel = scores.get("answer_relevancy")
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
    """Import metric classes defensively across RAGAS versions."""
    from ragas.metrics import Faithfulness

    try:
        from ragas.metrics import ResponseRelevancy as AnswerRel
    except ImportError:  # naming differs across versions
        from ragas.metrics import AnswerRelevancy as AnswerRel

    metrics = [Faithfulness(), AnswerRel()]
    if with_reference:
        try:
            from ragas.metrics import (
                LLMContextPrecisionWithReference,
                LLMContextRecall,
            )

            metrics += [LLMContextPrecisionWithReference(), LLMContextRecall()]
        except ImportError:
            pass
    return metrics


def _normalize(col: str):
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
    """Shim the removed ``langchain_community.chat_models.vertexai`` path.

    ragas 0.4.x imports ``ChatVertexAI`` (and ``VertexAI``) unconditionally at
    module load, but langchain-community 0.4+ dropped that submodule. We never
    use Vertex, so registering lightweight stubs lets ``import ragas`` succeed.
    Idempotent and only fills in what's actually missing.
    """
    import sys
    import types

    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name not in sys.modules:
        try:  # real module wins if a compatible version is installed
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
    """Evaluate one sample with RAGAS. Never raises; returns NaN + _error on failure.

    Context precision/recall run only when a reference is supplied.
    """
    base = {"faithfulness": float("nan"), "answer_relevancy": float("nan")}
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

        from ..config import JUDGE_MODEL, make_chat_llm
        from ..rag.vectorstore import get_embeddings

        # Faithfulness emits one NLI verdict + reason per extracted statement,
        # so its structured JSON is far larger than answer-relevancy's short
        # question list. A 1024 cap truncated that JSON → instructor parse
        # failure → NaN for faithfulness only. 4096 leaves ample headroom.
        judge = LangchainLLMWrapper(make_chat_llm(JUDGE_MODEL, max_tokens=4096))
        embeddings = LangchainEmbeddingsWrapper(get_embeddings())
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=list(contexts),
            reference=reference,
        )
        dataset = EvaluationDataset(samples=[sample])
        metrics = _load_metrics(with_reference=bool(reference))
        result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings)
        df = result.to_pandas()
        for col in df.columns:
            name = _normalize(col)
            if name is not None:
                try:
                    base[name] = float(df[col].mean())
                except (TypeError, ValueError):
                    continue
        return base
    except Exception as exc:  # keep the graph alive
        base["_error"] = str(exc)
        return base
