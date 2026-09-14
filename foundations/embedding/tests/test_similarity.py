"""
tests/test_similarity.py
-------------------------
Unit tests for similarity.py. Run with:
    pytest tests/test_similarity.py -v

These tests use plain numeric vectors (not real embeddings) to test the
math in isolation, plus one integration-style test using
SimpleHashEmbeddingProvider (offline, no downloads) to sanity-check the
full embed -> compare pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running `pytest` from the package root without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from similarity import (  # noqa: E402
    cosine_similarity,
    dot_product,
    euclidean_distance,
    euclidean_to_similarity,
    top_k_similar,
)
from embedding_demo import SimpleHashEmbeddingProvider  # noqa: E402


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_ignores_magnitude(self):
        a = [1.0, 1.0]
        b = [10.0, 10.0]  # same direction, 10x magnitude
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_zero_vector_returns_zero_not_error(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_accepts_numpy_arrays(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# dot_product
# ---------------------------------------------------------------------------

class TestDotProduct:
    def test_basic_dot_product(self):
        assert dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)

    def test_sensitive_to_magnitude(self):
        a = [1.0, 1.0]
        small = dot_product(a, [1.0, 1.0])
        large = dot_product(a, [10.0, 10.0])
        assert large > small

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            dot_product([1.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# euclidean_distance / euclidean_to_similarity
# ---------------------------------------------------------------------------

class TestEuclideanDistance:
    def test_identical_vectors_zero_distance(self):
        v = [1.0, 2.0, 3.0]
        assert euclidean_distance(v, v) == pytest.approx(0.0)

    def test_known_distance(self):
        a = [0.0, 0.0]
        b = [3.0, 4.0]  # classic 3-4-5 triangle
        assert euclidean_distance(a, b) == pytest.approx(5.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            euclidean_distance([1.0, 2.0], [1.0])

    def test_similarity_conversion_is_bounded_and_monotonic(self):
        near = euclidean_to_similarity(0.0)
        mid = euclidean_to_similarity(1.0)
        far = euclidean_to_similarity(100.0)
        assert near == pytest.approx(1.0)
        assert 0.0 < far < mid < near <= 1.0


# ---------------------------------------------------------------------------
# top_k_similar
# ---------------------------------------------------------------------------

class TestTopKSimilar:
    def setup_method(self):
        self.candidates = np.array(
            [
                [1.0, 0.0],   # idx 0: identical to query
                [0.9, 0.1],   # idx 1: very close
                [0.0, 1.0],   # idx 2: orthogonal
                [-1.0, 0.0],  # idx 3: opposite
            ]
        )
        self.query = [1.0, 0.0]

    def test_returns_k_results_sorted_descending(self):
        results = top_k_similar(self.query, self.candidates, k=3, metric="cosine")
        assert len(results) == 3
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_best_match_is_identical_vector(self):
        results = top_k_similar(self.query, self.candidates, k=1, metric="cosine")
        best_idx, best_score = results[0]
        assert best_idx == 0
        assert best_score == pytest.approx(1.0)

    def test_k_larger_than_candidates_is_clamped(self):
        results = top_k_similar(self.query, self.candidates, k=100, metric="cosine")
        assert len(results) == len(self.candidates)

    def test_empty_candidates_returns_empty_list(self):
        empty = np.zeros((0, 2))
        assert top_k_similar(self.query, empty, k=5) == []

    def test_euclidean_metric_orders_by_distance(self):
        results = top_k_similar(self.query, self.candidates, k=4, metric="euclidean")
        # nearest in euclidean space should still be the identical vector
        assert results[0][0] == 0

    def test_dot_metric_runs_without_error(self):
        results = top_k_similar(self.query, self.candidates, k=2, metric="dot")
        assert len(results) == 2

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            top_k_similar(self.query, self.candidates, k=1, metric="not_a_real_metric")

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            top_k_similar([1.0, 0.0, 0.0], self.candidates, k=1)


# ---------------------------------------------------------------------------
# Integration: SimpleHashEmbeddingProvider + cosine_similarity
# ---------------------------------------------------------------------------

class TestEmbeddingIntegration:
    def setup_method(self):
        self.provider = SimpleHashEmbeddingProvider(dimension=128)

    def test_identical_text_scores_one(self):
        text = "agentic AI uses tools"
        v1 = self.provider.embed_one(text)
        v2 = self.provider.embed_one(text)
        assert cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-5)

    def test_similar_text_scores_higher_than_unrelated_text(self):
        base = self.provider.embed_one("The cat sat on the mat")
        similar = self.provider.embed_one("The cat sits on the mat")
        unrelated = self.provider.embed_one("Quantum entanglement in particle physics")

        sim_score = cosine_similarity(base, similar)
        unrelated_score = cosine_similarity(base, unrelated)
        assert sim_score > unrelated_score

    def test_embeddings_are_correct_dimension(self):
        vectors = self.provider.embed(["hello world", "foo bar baz"])
        assert vectors.shape == (2, 128)

    def test_embeddings_are_normalized(self):
        vector = self.provider.embed_one("some example text")
        norm = np.linalg.norm(vector)
        assert norm == pytest.approx(1.0, abs=1e-5)
