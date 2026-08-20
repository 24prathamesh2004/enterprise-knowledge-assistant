
import logging
import math
import nest_asyncio
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig
from config import JUDGE_MODEL, EMBED_MODEL, make_llm
from rag import get_embeddings

# RAGAS renamed a few metrics across versions; support both.
try:
    from ragas.metrics import ResponseRelevancy as AnswerRelevancy
except ImportError:
    from ragas.metrics import AnswerRelevancy

try:
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
    HAS_REFERENCE_METRICS = True
except ImportError:
    HAS_REFERENCE_METRICS = False

nest_asyncio.apply()

METRIC_ORDER = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


class RAGASEmbeddings:
    """Adapt LocalEmbeddings to what RAGAS expects (it reads ``model_name``)."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.model_name = getattr(embeddings, "model_name", EMBED_MODEL)

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text):
        return self.embeddings.embed_query(text)

    def __str__(self):
        return self.model_name


def interpret_scores(scores: dict) -> str:
    """Turn raw RAGAS scores into plain-language lines plus a one-line verdict."""
    lines = []
    for metric in METRIC_ORDER:
        value = scores.get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            if metric in ("context_precision", "context_recall"):
                shown, label = "n/a", "n/a (requires reference answer)"
            else:
                shown, label = "nan", "n/a (evaluation unavailable)"
        elif value >= 0.80:
            shown, label = f"{value:.2f}", "strong"
        elif value >= 0.60:
            shown, label = f"{value:.2f}", "acceptable (minor gaps)"
        else:
            shown, label = f"{value:.2f}", "weak"
        lines.append(f"- {metric}: {shown} — {label}")

    faith = scores.get("faithfulness")
    relevancy = scores.get("answer_relevancy")
    faith_valid = faith is not None and not (isinstance(faith, float) and math.isnan(faith))
    rel_valid = relevancy is not None and not (isinstance(relevancy, float) and math.isnan(relevancy))

    if not faith_valid or not rel_valid:
        summary = "Evaluation incomplete — treat answer with caution."
    elif faith >= 0.70 and relevancy >= 0.70:
        summary = "Faithful and on-topic; answer is well grounded."
    elif faith < 0.70:
        summary = "Low faithfulness — answer may contain unsupported claims."
    else:
        summary = "Low relevancy — may not directly address the question."

    lines.append(f"Summary: {summary}")
    return "\n".join(lines)


def normalize_column_name(column: str) -> str:
    col = column.lower()
    if "faith" in col:
        return "faithfulness"
    if "relevanc" in col and "context" not in col:
        return "answer_relevancy"
    if "precision" in col:
        return "context_precision"
    if "recall" in col:
        return "context_recall"
    return None


def run_ragas(question: str, answer: str, contexts: list[str], reference: str = None) -> dict:
    scores = {
        "faithfulness": float("nan"),
        "answer_relevancy": float("nan"),
        "context_precision": float("nan"),
        "context_recall": float("nan"),
    }

    try:
        # Faithfulness and answer relevancy need no reference; add the context
        # metrics only when a golden-set reference is supplied.
        metrics = [Faithfulness(), AnswerRelevancy()]
        if reference and HAS_REFERENCE_METRICS:
            metrics.extend([LLMContextPrecisionWithReference(), LLMContextRecall()])

        judge = LangchainLLMWrapper(make_llm(JUDGE_MODEL, max_tokens=4096), bypass_n=True)
        embeddings = LangchainEmbeddingsWrapper(RAGASEmbeddings(get_embeddings()))

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=list(contexts),
            reference=reference,
        )
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=metrics,
            llm=judge,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=1, timeout=120, max_retries=2),
        )

        df = result.to_pandas()
        for column in df.columns:
            metric_name = normalize_column_name(column)
            if metric_name in scores:
                value = float(df[column].mean())
                if not math.isnan(value):
                    scores[metric_name] = value
    except Exception as e:
        logging.getLogger(__name__).warning("RAGAS evaluation failed: %s: %s", type(e).__name__, e)

    return scores
