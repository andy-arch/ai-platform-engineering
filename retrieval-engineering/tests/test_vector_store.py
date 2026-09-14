"""
tests/test_vector_store.py
-----------------------------
Unit tests for vector_store/local_store.py: add/search ranking, save/load
round-tripping, metadata filtering, and edge cases. Uses small synthetic
vectors (not real embeddings) so similarity ordering is exactly known.
"""

from pathlib import Path

import numpy as np
import pytest

from retrieval_engineering.models.document import Chunk
from retrieval_engineering.vector_store.local_store import LocalVectorStore


def _chunk(chunk_id: str, source: str = "a/b.md", **metadata) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=f"text for {chunk_id}",
        source=source,
        chunk_index=0,
        metadata=metadata,
    )


class TestAdd:
    def test_add_and_len(self):
        store = LocalVectorStore()
        chunks = [_chunk("c1"), _chunk("c2")]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store.add(chunks, vectors)
        assert len(store) == 2

    def test_mismatched_counts_raise(self):
        store = LocalVectorStore()
        chunks = [_chunk("c1"), _chunk("c2")]
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            store.add(chunks, vectors)

    def test_duplicate_chunk_id_raises(self):
        store = LocalVectorStore()
        store.add([_chunk("c1")], np.array([[1.0, 0.0]], dtype=np.float32))
        with pytest.raises(ValueError):
            store.add([_chunk("c1")], np.array([[0.0, 1.0]], dtype=np.float32))

    def test_multiple_add_calls_accumulate(self):
        store = LocalVectorStore()
        store.add([_chunk("c1")], np.array([[1.0, 0.0]], dtype=np.float32))
        store.add([_chunk("c2")], np.array([[0.0, 1.0]], dtype=np.float32))
        assert len(store) == 2


class TestSearch:
    def test_returns_closest_vector_first(self):
        store = LocalVectorStore()
        chunks = [_chunk("close"), _chunk("far"), _chunk("opposite")]
        vectors = np.array([[0.99, 0.01], [0.1, 0.99], [-1.0, 0.0]], dtype=np.float32)
        store.add(chunks, vectors)

        results = store.search(np.array([1.0, 0.0]), k=3)
        assert [r.chunk.chunk_id for r in results] == ["close", "far", "opposite"]

    def test_respects_k(self):
        store = LocalVectorStore()
        chunks = [_chunk(f"c{i}") for i in range(5)]
        vectors = np.random.RandomState(0).rand(5, 2).astype(np.float32)
        store.add(chunks, vectors)
        results = store.search(np.array([1.0, 1.0]), k=2)
        assert len(results) == 2

    def test_empty_store_returns_empty_list(self):
        store = LocalVectorStore()
        assert store.search(np.array([1.0, 0.0]), k=5) == []

    def test_scores_sorted_descending(self):
        store = LocalVectorStore()
        chunks = [_chunk(f"c{i}") for i in range(4)]
        vectors = np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        store.add(chunks, vectors)
        results = store.search(np.array([1.0, 0.0]), k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestMetadataFiltering:
    def test_filter_restricts_results(self):
        store = LocalVectorStore()
        chunks = [
            _chunk("aws1", category="aws"),
            _chunk("k8s1", category="kubernetes"),
        ]
        vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        store.add(chunks, vectors)

        results = store.search(np.array([1.0, 0.0]), k=5, filters={"category": "aws"})
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "aws1"

    def test_filter_matching_nothing_returns_empty(self):
        store = LocalVectorStore()
        store.add([_chunk("c1", category="aws")], np.array([[1.0, 0.0]], dtype=np.float32))
        results = store.search(np.array([1.0, 0.0]), k=5, filters={"category": "nonexistent"})
        assert results == []


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        store = LocalVectorStore()
        chunks = [_chunk("c1", category="aws"), _chunk("c2", category="kubernetes")]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store.add(chunks, vectors)

        index_path = tmp_path / "kb"
        store.save(index_path)

        assert Path(f"{index_path}.vectors.npy").exists()
        assert Path(f"{index_path}.payload.json").exists()

        loaded = LocalVectorStore.load(index_path)
        assert len(loaded) == 2

        results = loaded.search(np.array([1.0, 0.0]), k=1)
        assert results[0].chunk.chunk_id == "c1"

    def test_loaded_store_preserves_metadata(self, tmp_path: Path):
        store = LocalVectorStore()
        store.add([_chunk("c1", category="aws", topic="eks")], np.array([[1.0, 0.0]], dtype=np.float32))
        index_path = tmp_path / "kb"
        store.save(index_path)

        loaded = LocalVectorStore.load(index_path)
        assert loaded.search(np.array([1.0, 0.0]), k=1)[0].chunk.metadata["topic"] == "eks"
