"""
tests/test_embeddings.py
--------------------------
Unit tests for embeddings/embedder.py. Uses SimpleHashEmbeddingProvider
explicitly (offline, deterministic, no model download) rather than
get_default_provider(), so this suite never needs network access -- same
convention as foundations/embedding's own test suite.
"""

import numpy as np

from foundations_embedding.embedding_demo import SimpleHashEmbeddingProvider
from retrieval_engineering.embeddings.embedder import Embedder


def _embedder() -> Embedder:
    return Embedder(provider=SimpleHashEmbeddingProvider(dimension=64))


class TestEmbedText:
    def test_returns_vector_of_correct_dimension(self):
        embedder = _embedder()
        vector = embedder.embed_text("hello world")
        assert vector.shape == (64,)

    def test_is_deterministic(self):
        embedder = _embedder()
        v1 = embedder.embed_text("hello world")
        v2 = embedder.embed_text("hello world")
        assert np.array_equal(v1, v2)

    def test_different_text_generally_differs(self):
        embedder = _embedder()
        v1 = embedder.embed_text("cats and dogs")
        v2 = embedder.embed_text("quantum mechanics")
        assert not np.array_equal(v1, v2)


class TestEmbedDocuments:
    def test_returns_matrix_of_correct_shape(self):
        embedder = _embedder()
        vectors = embedder.embed_documents(["one", "two", "three"])
        assert vectors.shape == (3, 64)

    def test_matches_embed_text_row_by_row(self):
        embedder = _embedder()
        texts = ["alpha", "beta"]
        batch = embedder.embed_documents(texts)
        individual = np.stack([embedder.embed_text(t) for t in texts])
        assert np.allclose(batch, individual)

    def test_empty_batch_returns_empty_matrix(self):
        embedder = _embedder()
        vectors = embedder.embed_documents([])
        assert vectors.shape[0] == 0


class TestDimensionProperty:
    def test_dimension_matches_provider(self):
        embedder = Embedder(provider=SimpleHashEmbeddingProvider(dimension=128))
        assert embedder.dimension == 128
