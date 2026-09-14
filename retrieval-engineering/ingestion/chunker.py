"""
ingestion/chunker.py
---------------------
Splits a Document's content into overlapping Chunks of roughly
`chunk_size` characters, trying not to split a word in half.

Trade-offs (also covered in the top-level README):

- Small chunks: retrieval is more precise -- each hit contains less
  irrelevant surrounding text -- but you lose context that spans a
  boundary, and you multiply the number of vectors to store, index and
  search (and the number of embedding calls at ingestion time).
- Large chunks: preserve more context per hit and mean fewer vectors to
  manage, but dilute relevance -- a chunk that's half about the query
  topic and half about something else scores lower than a tightly-scoped
  chunk would, and a downstream generation step has more irrelevant text
  to read through.
- Overlap: repeats a little text across the boundary between consecutive
  chunks, so a concept that would otherwise be split across two chunks
  is still fully readable from at least one of them. Costs extra storage
  and embedding compute proportional to the overlap fraction.

This module implements one strategy (fixed-size, character-based,
word-boundary-aware). `ChunkingStrategy` is a Protocol so token-aware or
structure-aware (e.g. split-on-heading) strategies can be added later
without changing any caller.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Protocol

from retrieval_engineering.models.document import Chunk, Document


class ChunkingStrategy(Protocol):
    def chunk(self, document: Document) -> List[Chunk]: ...


def _chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """
    Deterministic, content-addressable chunk id: same document + same
    index + same text always produces the same id, across runs and
    machines. Also means re-ingesting unchanged content is a no-op at
    the id level, and a changed chunk gets a new id rather than silently
    reusing a stale one.
    """
    digest = hashlib.sha256(f"{document_id}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]


class FixedSizeChunker:
    """
    Fixed-size character chunker with configurable overlap.

    Grows a chunk up to `chunk_size` characters, then backs off to the
    last whitespace boundary so words aren't split in half (falls back to
    a hard cut only if a single "word" is itself longer than chunk_size).
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        pieces: List[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= length:
                break

            next_start = max(end - self.overlap, start + 1)  # guarantee forward progress
            if next_start < end:
                # back off to a whitespace boundary so the next chunk doesn't
                # start mid-word; search forward first (keeps closer to the
                # requested overlap), then backward as a fallback.
                forward = text.find(" ", next_start, end)
                if forward != -1:
                    next_start = forward + 1
                else:
                    backward = text.rfind(" ", start, next_start)
                    if backward > start:
                        next_start = backward + 1
            start = next_start

        return pieces

    def chunk(self, document: Document) -> List[Chunk]:
        texts = self._split_text(document.content)
        chunks: List[Chunk] = []
        for index, text in enumerate(texts):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(document.id, index, text),
                    document_id=document.id,
                    text=text,
                    source=document.source,
                    chunk_index=index,
                    metadata=dict(document.metadata),
                )
            )
        return chunks


def chunk_documents(
    documents: List[Document], strategy: Optional[ChunkingStrategy] = None
) -> List[Chunk]:
    """Chunk a batch of documents with the given strategy (default: FixedSizeChunker())."""
    strategy = strategy or FixedSizeChunker()
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(strategy.chunk(document))
    return chunks
