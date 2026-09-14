"""
tests/test_evaluation.py
---------------------------
Unit tests for evaluation/metrics.py: Recall@K, MRR, and the evaluate()
aggregation, using synthetic RetrievalResult lists so the expected
numbers are known exactly.
"""

from retrieval_engineering.evaluation.metrics import evaluate, recall_at_k, reciprocal_rank
from retrieval_engineering.models.document import Chunk, RetrievalResult


def _result(source: str, score: float) -> RetrievalResult:
    chunk = Chunk(chunk_id=source, document_id="d", text="t", source=source, chunk_index=0)
    return RetrievalResult(chunk=chunk, score=score)


class TestRecallAtK:
    def test_hit_within_k(self):
        results = [_result("a.md", 0.9), _result("b.md", 0.8), _result("c.md", 0.7)]
        assert recall_at_k(results, expected_sources=["b.md"], k=3) == 1.0

    def test_miss_outside_k(self):
        results = [_result("a.md", 0.9), _result("b.md", 0.8), _result("c.md", 0.7)]
        assert recall_at_k(results, expected_sources=["c.md"], k=2) == 0.0

    def test_no_relevant_results_at_all(self):
        results = [_result("a.md", 0.9)]
        assert recall_at_k(results, expected_sources=["z.md"], k=1) == 0.0


class TestReciprocalRank:
    def test_first_result_is_relevant(self):
        results = [_result("a.md", 0.9), _result("b.md", 0.8)]
        assert reciprocal_rank(results, expected_sources=["a.md"]) == 1.0

    def test_second_result_is_relevant(self):
        results = [_result("a.md", 0.9), _result("b.md", 0.8)]
        assert reciprocal_rank(results, expected_sources=["b.md"]) == 0.5

    def test_no_relevant_result_scores_zero(self):
        results = [_result("a.md", 0.9), _result("b.md", 0.8)]
        assert reciprocal_rank(results, expected_sources=["z.md"]) == 0.0


class TestEvaluate:
    def test_aggregates_recall_and_mrr_across_questions(self):
        questions = [
            {"question": "q1", "expected_sources": ["a.md"]},
            {"question": "q2", "expected_sources": ["missing.md"]},
        ]

        def retrieve_fn(question: str, k: int):
            if question == "q1":
                return [_result("a.md", 0.9), _result("b.md", 0.8)]
            return [_result("x.md", 0.5)]

        report = evaluate(questions, retrieve_fn, k_values=[1, 3])

        assert report["summary"]["num_questions"] == 2
        assert report["summary"]["recall@1"] == 0.5  # q1 hits, q2 misses
        assert report["summary"]["mrr"] == 0.5  # (1.0 + 0.0) / 2

    def test_empty_question_set(self):
        report = evaluate([], retrieve_fn=lambda q, k: [], k_values=[1])
        assert report["summary"]["num_questions"] == 0
        assert report["summary"]["mrr"] == 0.0
