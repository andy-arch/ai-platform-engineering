"""
tests/test_integration.py
----------------------------
End-to-end integration test:

    documents -> ingestion -> chunking -> embeddings -> vector store -> query -> retrieval

Builds a tiny temporary knowledge base (two categories), runs the full
pipeline with SimpleHashEmbeddingProvider (offline, deterministic), and
checks that a query using vocabulary from a specific document retrieves
that document's chunk. Also exercises save/load of the vector store as
part of the same flow.
"""

from pathlib import Path

from foundations_embedding.embedding_demo import SimpleHashEmbeddingProvider
from retrieval_engineering.embeddings.embedder import Embedder
from retrieval_engineering.ingestion.chunker import FixedSizeChunker, chunk_documents
from retrieval_engineering.ingestion.loader import load_documents
from retrieval_engineering.retrieval.semantic_retriever import SemanticRetriever
from retrieval_engineering.vector_store.local_store import LocalVectorStore


def _write_knowledge_base(root: Path) -> None:
    (root / "aws").mkdir(parents=True)
    (root / "kubernetes").mkdir(parents=True)

    (root / "aws" / "s3.md").write_text(
        "# Amazon S3\n\n"
        "Amazon S3 is an object store. Objects are stored in buckets and "
        "identified by a key, with strong read after write consistency.",
        encoding="utf-8",
    )
    (root / "kubernetes" / "rbac.md").write_text(
        "# Kubernetes RBAC\n\n"
        "Kubernetes RBAC controls access with Roles and RoleBindings, "
        "granting verbs like get list and create on cluster resources.",
        encoding="utf-8",
    )


class TestFullPipeline:
    def test_ingest_then_retrieve_finds_the_right_document(self, tmp_path: Path):
        knowledge_dir = tmp_path / "knowledge"
        _write_knowledge_base(knowledge_dir)

        # ingestion -> chunking
        documents = load_documents(knowledge_dir)
        assert len(documents) == 2
        chunks = chunk_documents(documents, FixedSizeChunker(chunk_size=200, overlap=20))
        assert len(chunks) >= 2

        # embeddings -> vector store
        embedder = Embedder(provider=SimpleHashEmbeddingProvider(dimension=128))
        vectors = embedder.embed_documents([c.text for c in chunks])
        store = LocalVectorStore()
        store.add(chunks, vectors)

        # persist + reload, to exercise the same path scripts/ingest.py and
        # scripts/search.py use across two separate processes
        index_path = tmp_path / "index" / "kb"
        store.save(index_path)
        reloaded_store = LocalVectorStore.load(index_path)

        # query -> retrieval
        retriever = SemanticRetriever(embedder=embedder, vector_store=reloaded_store)
        results = retriever.retrieve("Roles and RoleBindings control access to cluster resources", top_k=2)

        assert len(results) > 0
        assert results[0].chunk.source == "kubernetes/rbac.md"
        assert results[0].chunk.metadata["category"] == "kubernetes"

    def test_metadata_filter_scopes_retrieval_to_one_category(self, tmp_path: Path):
        knowledge_dir = tmp_path / "knowledge"
        _write_knowledge_base(knowledge_dir)

        documents = load_documents(knowledge_dir)
        chunks = chunk_documents(documents, FixedSizeChunker(chunk_size=200, overlap=20))
        embedder = Embedder(provider=SimpleHashEmbeddingProvider(dimension=128))
        vectors = embedder.embed_documents([c.text for c in chunks])
        store = LocalVectorStore()
        store.add(chunks, vectors)
        retriever = SemanticRetriever(embedder=embedder, vector_store=store)

        results = retriever.retrieve(
            "Roles and RoleBindings control access to cluster resources",
            top_k=5,
            filters={"category": "aws"},
        )

        assert all(r.chunk.metadata["category"] == "aws" for r in results)
