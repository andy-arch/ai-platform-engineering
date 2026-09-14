"""
vector_store/local_store.py
------------------------------
A local, NumPy-backed vector store: the mechanics underneath a "vector
database" reduced to their essentials -- a matrix of vectors, a parallel
list of payloads (chunk text + metadata), and brute-force similarity
search over that matrix via foundations_embedding.similarity.top_k_similar.

Persistence is two files rather than one, deliberately mirroring how real
vector databases separate vector data from payload:

    <path>.vectors.npy  -- the (n, dim) float32 embedding matrix
    <path>.payload.json -- chunk ids/text/source/metadata, row-aligned
                            with the matrix (row i of the matrix <->
                            payload["chunks"][i])

This keeps embeddings out of JSON (slow to parse, bloats the file for a
large corpus) while keeping the payload itself human-inspectable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from foundations_embedding.similarity import top_k_similar
from retrieval_engineering.models.document import Chunk, RetrievalResult

logger = logging.getLogger(__name__)

MetadataFilter = Dict[str, Any]


class LocalVectorStore:
    """
    Brute-force local vector store.

    Example:
        store = LocalVectorStore()
        store.add(chunks, vectors)
        store.save("data/index/kb")
        ...
        store2 = LocalVectorStore.load("data/index/kb")
        results = store2.search(query_vector, k=5)
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self._chunks: List[Chunk] = []
        self._vectors: Optional[np.ndarray] = None  # shape (n, dim)
        self._id_to_index: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """Add chunks and their precomputed embeddings. vectors: shape (len(chunks), dim)."""
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"chunk/vector count mismatch: {len(chunks)} vs {vectors.shape[0]}")

        duplicates = {c.chunk_id for c in chunks} & set(self._id_to_index.keys())
        if duplicates:
            raise ValueError(f"chunk_id(s) already exist in store: {sorted(duplicates)}")

        vectors = np.asarray(vectors, dtype=np.float32)

        for chunk in chunks:
            self._id_to_index[chunk.chunk_id] = len(self._chunks)
            self._chunks.append(chunk)

        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        logger.info("Added %d chunk(s); store now holds %d", len(chunks), len(self._chunks))

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: MetadataFilter) -> bool:
        return all(chunk.metadata.get(key) == value for key, value in filters.items())

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filters: Optional[MetadataFilter] = None,
    ) -> List[RetrievalResult]:
        """
        Return the top-k chunks most similar to query_vector, optionally
        restricted to chunks whose metadata matches every key/value pair
        in `filters` (e.g. {"category": "aws"}).
        """
        if self._vectors is None or len(self._chunks) == 0:
            return []

        if filters:
            candidate_indices = [
                i for i, chunk in enumerate(self._chunks) if self._matches_filters(chunk, filters)
            ]
            if not candidate_indices:
                return []
            candidate_matrix = self._vectors[candidate_indices]
        else:
            candidate_indices = list(range(len(self._chunks)))
            candidate_matrix = self._vectors

        ranked = top_k_similar(query_vector, candidate_matrix, k=k, metric=self.metric)
        return [
            RetrievalResult(chunk=self._chunks[candidate_indices[local_idx]], score=score)
            for local_idx, score in ranked
        ]

    def save(self, path: str | Path) -> None:
        """Persist to <path>.vectors.npy + <path>.payload.json."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        vectors = self._vectors if self._vectors is not None else np.zeros((0, 0), dtype=np.float32)
        np.save(f"{path}.vectors.npy", vectors)

        payload = {"metric": self.metric, "chunks": [c.model_dump() for c in self._chunks]}
        Path(f"{path}.payload.json").write_text(json.dumps(payload, indent=2))
        logger.info("Saved %d chunk(s) to %s.{vectors.npy,payload.json}", len(self._chunks), path)

    @classmethod
    def load(cls, path: str | Path) -> "LocalVectorStore":
        """Load a store previously written by save()."""
        path = Path(path)
        vectors = np.load(f"{path}.vectors.npy")
        payload = json.loads(Path(f"{path}.payload.json").read_text())

        store = cls(metric=payload.get("metric", "cosine"))
        store._vectors = vectors if vectors.size else None
        store._chunks = [Chunk(**c) for c in payload["chunks"]]
        store._id_to_index = {c.chunk_id: i for i, c in enumerate(store._chunks)}
        return store
