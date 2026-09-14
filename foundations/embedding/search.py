"""
search.py
---------
A minimal in-memory vector search engine, built on top of embedding_demo.py
and similarity.py. This is the "toy retriever" that RAG pipelines and
agentic tool-use (e.g. a `search_knowledge_base` tool) are built on top of.

Design:
- Documents are stored with an id, text, optional metadata, and their
  precomputed embedding.
- `add_documents` embeds and stores in a batch (batching embedding calls
  is what makes this fast/cheap in real systems -- avoid embedding one
  document at a time in a loop).
- `search` embeds the query once and ranks all stored vectors.
- Index can be persisted to / loaded from disk as JSON (embeddings are
  stored as plain lists so the file is portable and human-inspectable).

This is intentionally O(n) brute-force search -- fine for thousands of
documents, and a deliberate baseline before reaching for an ANN index
(FAISS, HNSW, a managed vector DB, etc.) in a later module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from embedding_demo import EmbeddingProvider, get_default_provider
from similarity import top_k_similar


@dataclass
class Document:
    """A single indexed document."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "Document":
        embedding = np.array(data["embedding"], dtype=np.float32) if data.get("embedding") else None
        return cls(id=data["id"], text=data["text"], metadata=data.get("metadata", {}), embedding=embedding)


@dataclass
class SearchResult:
    """A single scored search hit."""

    document: Document
    score: float

    def __repr__(self) -> str:
        preview = self.document.text[:60] + ("..." if len(self.document.text) > 60 else "")
        return f"SearchResult(id={self.document.id!r}, score={self.score:.4f}, text={preview!r})"


class VectorSearchEngine:
    """
    A brute-force, in-memory semantic search index.

    Example:
        engine = VectorSearchEngine()
        engine.add_documents([
            {"id": "1", "text": "Agentic AI plans and uses tools."},
            {"id": "2", "text": "Embeddings turn text into vectors."},
        ])
        results = engine.search("what is an embedding?", k=1)
    """

    def __init__(self, provider: Optional[EmbeddingProvider] = None, metric: str = "cosine"):
        self.provider = provider or get_default_provider()
        self.metric = metric
        self._documents: List[Document] = []
        self._id_to_index: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._documents)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add a batch of documents.

        Args:
            documents: list of dicts each with keys:
                - "id" (str, required, must be unique in this index)
                - "text" (str, required)
                - "metadata" (dict, optional)
        """
        if not documents:
            return

        new_ids = [doc["id"] for doc in documents]
        duplicates = set(new_ids) & set(self._id_to_index.keys())
        if duplicates:
            raise ValueError(f"Document id(s) already exist in index: {sorted(duplicates)}")
        if len(new_ids) != len(set(new_ids)):
            raise ValueError("Duplicate ids found within the batch being added.")

        texts = [doc["text"] for doc in documents]
        vectors = self.provider.embed(texts)  # batched embedding call

        for doc, vector in zip(documents, vectors):
            record = Document(
                id=doc["id"],
                text=doc["text"],
                metadata=doc.get("metadata", {}),
                embedding=vector,
            )
            self._id_to_index[record.id] = len(self._documents)
            self._documents.append(record)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document by id and rebuild the internal index map."""
        if doc_id not in self._id_to_index:
            raise KeyError(f"No document with id '{doc_id}' in index.")
        idx = self._id_to_index[doc_id]
        del self._documents[idx]
        self._id_to_index = {doc.id: i for i, doc in enumerate(self._documents)}

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Embed the query and return the top-k most similar documents."""
        if len(self._documents) == 0:
            return []

        query_vector = self.provider.embed_one(query)
        matrix = np.stack([doc.embedding for doc in self._documents])

        ranked = top_k_similar(query_vector, matrix, k=k, metric=self.metric)
        return [SearchResult(document=self._documents[idx], score=score) for idx, score in ranked]

    def save(self, path: str | Path) -> None:
        """Persist the index (documents + embeddings) to a JSON file."""
        path = Path(path)
        payload = {
            "metric": self.metric,
            "provider": self.provider.__class__.__name__,
            "documents": [doc.to_json_dict() for doc in self._documents],
        }
        path.write_text(json.dumps(payload, indent=2))

    def load(self, path: str | Path) -> None:
        """
        Load documents + precomputed embeddings from a JSON file previously
        written by `save()`. Does NOT re-embed -- if you swap providers,
        old embeddings won't be comparable to new query embeddings, so
        re-index from source text instead in that case.
        """
        path = Path(path)
        payload = json.loads(path.read_text())
        self.metric = payload.get("metric", self.metric)
        self._documents = [Document.from_json_dict(d) for d in payload["documents"]]
        self._id_to_index = {doc.id: i for i, doc in enumerate(self._documents)}


def _demo() -> None:
    engine = VectorSearchEngine()
    engine.add_documents(
        [
            {"id": "1", "text": "Agentic AI systems can plan, reason, and call tools autonomously."},
            {"id": "2", "text": "Embeddings represent text as dense numerical vectors."},
            {"id": "3", "text": "The stock market saw significant volatility this week."},
            {"id": "4", "text": "Cosine similarity measures the angle between two vectors."},
            {"id": "5", "text": "A recipe for sourdough bread requires flour, water, and patience."},
        ]
    )

    query = "how do vectors capture meaning?"
    print(f"Query: {query!r}\n")
    for result in engine.search(query, k=3):
        print(result)


if __name__ == "__main__":
    _demo()
