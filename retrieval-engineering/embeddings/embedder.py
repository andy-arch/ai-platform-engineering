"""
embeddings/embedder.py
------------------------
Thin adapter around Phase 1's EmbeddingProvider (foundations/embedding),
exposing the embed_text() / embed_documents() interface used throughout
this retrieval pipeline.

Reusing Phase 1's provider -- rather than reimplementing embedding logic
here -- means swapping SimpleHashEmbeddingProvider for
SentenceTransformerEmbeddingProvider or OpenAIEmbeddingProvider is a
one-line change (pass a different `provider` in), and nothing in
ingestion, chunking, the vector store, or retrieval needs to change.
That's the point of depending on the EmbeddingProvider abstraction rather
than a concrete provider.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from foundations_embedding.embedding_demo import EmbeddingProvider, get_default_provider


class Embedder:
    """Wraps an EmbeddingProvider with the naming convention this pipeline uses."""

    def __init__(self, provider: Optional[EmbeddingProvider] = None):
        self.provider = provider or get_default_provider()

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string -> 1D vector of shape (dimension,)."""
        return self.provider.embed_one(text)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of strings -> 2D array of shape (len(texts), dimension)."""
        return self.provider.embed(list(texts))
