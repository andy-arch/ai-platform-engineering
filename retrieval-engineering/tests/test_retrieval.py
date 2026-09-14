"""
tests/test_retrieval.py
--------------------------
Unit tests for retrieval/semantic_retriever.py. Uses SimpleHashEmbeddingProvider
so results are deterministic and offline.
"""

import pytest

from foundations_embedding.embedding_demo import SimpleHashEmbeddingProvider
from retrieval_engineering.embeddings.embedder import Embedder
from retrieval_engineering.ingestion.chunker import FixedSizeChunker, chunk_documents
from retrieval_engineering.models.document import Document
from retrieval_engineering.retrieval.semantic_retriever import SemanticRetriever
from retrieval_engineering.vector_store.local_store import LocalVectorStore


def _build_retriever() -> SemanticRetriever:
    documents = [
        Document(
            id="d1",
            source="kubernetes/rbac.md",
            filename="rbac.md",
            document_type="markdown",
            content="Kubernetes RBAC controls access using roles and role bindings for cluster permissions.",
            metadata={"category": "kubernetes", "topic": "rbac", "document_type": "markdown"},
        ),
        Document(
            id="d2",
            source="aws/s3.md",
            filename="s3.md",
            document_type="markdown",
            content="Amazon S3 is an object store for buckets and keys with strong read after write consistency.",
            metadata={"category": "aws", "topic": "s3", "document_type": "markdown"},
        ),
    ]
    chunks = chunk_documents(documents, FixedSizeChunker(chunk_size=500, overlap=50))
    embedder = Embedder(provider=SimpleHashEmbeddingProvider(dimension=128))
    vectors = embedder.embed_documents([c.text for c in chunks])

    store = LocalVectorStore()
    store.add(chunks, vectors)
    return SemanticRetriever(embedder=embedder, vector_store=store)


class TestRetrieve:
    def test_returns_results_sorted_by_score(self):
        retriever = _build_retriever()
        results = retriever.retrieve("roles and role bindings for cluster permissions", top_k=2)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_lexically_matching_query_ranks_its_source_first(self):
        retriever = _build_retriever()
        results = retriever.retrieve("Kubernetes RBAC controls access using roles", top_k=1)
        assert results[0].chunk.source == "kubernetes/rbac.md"

    def test_respects_top_k(self):
        retriever = _build_retriever()
        results = retriever.retrieve("object store buckets", top_k=1)
        assert len(results) == 1

    def test_empty_query_raises(self):
        retriever = _build_retriever()
        with pytest.raises(ValueError):
            retriever.retrieve("   ")

    def test_filters_are_passed_through_to_the_store(self):
        retriever = _build_retriever()
        results = retriever.retrieve("roles and permissions", top_k=5, filters={"category": "aws"})
        assert all(r.chunk.metadata["category"] == "aws" for r in results)
