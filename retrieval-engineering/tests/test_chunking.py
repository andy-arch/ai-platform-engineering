"""
tests/test_chunking.py
------------------------
Unit tests for ingestion/chunker.py: chunk sizing, overlap, deterministic
ids, word-boundary awareness, and metadata inheritance.
"""

from retrieval_engineering.ingestion.chunker import FixedSizeChunker, chunk_documents
from retrieval_engineering.models.document import Document


def _doc(content: str, doc_id: str = "doc1", source: str = "a/b.md") -> Document:
    return Document(
        id=doc_id,
        source=source,
        filename="b.md",
        document_type="markdown",
        content=content,
        metadata={"category": "a", "topic": "b", "document_type": "markdown"},
    )


class TestFixedSizeChunkerConfig:
    def test_rejects_non_positive_chunk_size(self):
        import pytest

        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=0)

    def test_rejects_overlap_too_large(self):
        import pytest

        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)


class TestChunkSizeAndOverlap:
    def test_short_document_produces_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk(_doc("A short document."))
        assert len(chunks) == 1
        assert chunks[0].text == "A short document."

    def test_long_document_produces_multiple_chunks_within_size_bound(self):
        text = " ".join(f"word{i}" for i in range(500))  # long text, plain words
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(_doc(text))

        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 100

    def test_consecutive_chunks_overlap(self):
        text = " ".join(f"word{i}" for i in range(200))
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(_doc(text))

        # some suffix of chunk[i] should reappear as a prefix-ish part of chunk[i+1]
        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())
        assert first_words & second_words

    def test_does_not_split_a_word_when_a_boundary_exists(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(_doc(text))

        all_words = set(text.split())
        for c in chunks:
            for token in c.text.split():
                assert token in all_words


class TestDeterministicChunkIds:
    def test_same_document_chunked_twice_yields_same_ids(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunker = FixedSizeChunker(chunk_size=80, overlap=10)

        ids1 = [c.chunk_id for c in chunker.chunk(_doc(text))]
        ids2 = [c.chunk_id for c in chunker.chunk(_doc(text))]

        assert ids1 == ids2

    def test_chunk_ids_are_unique_within_a_document(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunker = FixedSizeChunker(chunk_size=80, overlap=10)
        ids = [c.chunk_id for c in chunker.chunk(_doc(text))]
        assert len(ids) == len(set(ids))

    def test_different_content_yields_different_ids(self):
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        ids_a = {c.chunk_id for c in chunker.chunk(_doc("content A"))}
        ids_b = {c.chunk_id for c in chunker.chunk(_doc("content B"))}
        assert ids_a.isdisjoint(ids_b)


class TestChunkMetadata:
    def test_chunk_inherits_document_metadata_and_source(self):
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk(_doc("Some content.", source="aws/eks.md"))
        assert chunks[0].source == "aws/eks.md"
        assert chunks[0].metadata["category"] == "a"

    def test_chunk_index_increments(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunker = FixedSizeChunker(chunk_size=80, overlap=10)
        chunks = chunker.chunk(_doc(text))
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


class TestChunkDocuments:
    def test_chunks_multiple_documents(self):
        docs = [_doc("First document.", doc_id="d1"), _doc("Second document.", doc_id="d2")]
        chunks = chunk_documents(docs)
        assert {c.document_id for c in chunks} == {"d1", "d2"}
