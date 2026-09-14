"""
evaluation/metrics.py
-----------------------
Basic retrieval evaluation: Recall@K and Mean Reciprocal Rank (MRR)
against a small hand-labeled question set (evaluation/retrieval_questions.json).

Recall@K: for a single question, 1.0 if any of its expected sources
appears among the top-K retrieved chunks' sources, else 0.0. Averaged
over all questions, this answers "what fraction of the time did we find
a relevant document within the first K results?"

MRR (Mean Reciprocal Rank): for a single question, 1 / rank of the first
retrieved chunk whose source is in the expected set (0 if none of the
retrieved results match). Averaged over all questions. Unlike Recall@K,
MRR rewards ranking a relevant result higher -- rank 1 scores 1.0, rank 2
scores 0.5, rank 5 scores 0.2, and so on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from retrieval_engineering.models.document import RetrievalResult


def recall_at_k(results: List[RetrievalResult], expected_sources: List[str], k: int) -> float:
    top_sources = {r.chunk.source for r in results[:k]}
    return 1.0 if top_sources & set(expected_sources) else 0.0


def reciprocal_rank(results: List[RetrievalResult], expected_sources: List[str]) -> float:
    expected = set(expected_sources)
    for rank, result in enumerate(results, start=1):
        if result.chunk.source in expected:
            return 1.0 / rank
    return 0.0


def load_questions(path: str | Path) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text())


def evaluate(
    questions: List[Dict[str, Any]],
    retrieve_fn: Callable[[str, int], List[RetrievalResult]],
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, Any]:
    """
    Run every question through `retrieve_fn(question, top_k)` and compute
    Recall@K for each k in k_values, plus MRR, averaged over all questions.

    Returns {"summary": {...averaged metrics...}, "per_question": [...]}.
    """
    max_k = max(k_values) if k_values else 5
    per_question: List[Dict[str, Any]] = []

    for item in questions:
        question = item["question"]
        expected = item["expected_sources"]
        results = retrieve_fn(question, max_k)

        row: Dict[str, Any] = {
            "question": question,
            "expected_sources": expected,
            "retrieved_sources": [r.chunk.source for r in results],
            "reciprocal_rank": reciprocal_rank(results, expected),
        }
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(results, expected, k)
        per_question.append(row)

    n = len(per_question)
    summary: Dict[str, Any] = {
        "num_questions": n,
        "mrr": sum(r["reciprocal_rank"] for r in per_question) / n if n else 0.0,
    }
    for k in k_values:
        key = f"recall@{k}"
        summary[key] = sum(r[key] for r in per_question) / n if n else 0.0

    return {"summary": summary, "per_question": per_question}
