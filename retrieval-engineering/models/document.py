"""
models/document.py
-------------------
Core data models shared across the retrieval pipeline: a raw ingested
Document, the Chunk units produced from it, and a single scored
RetrievalResult. Pydantic gives us validation and one source of truth
for the shape of data flowing through
ingestion -> chunking -> embedding -> vector store -> retrieval.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single raw document loaded from the knowledge base, before chunking."""

    id: str
    source: str  # path relative to the knowledge root, e.g. "aws/eks.md"
    filename: str
    document_type: str  # "markdown" | "text"
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A single retrievable unit produced by chunking a Document."""

    chunk_id: str
    document_id: str
    text: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """A single scored retrieval hit, returned by the semantic retriever."""

    chunk: Chunk
    score: float
