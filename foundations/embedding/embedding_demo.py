"""
embedding_demo.py
------------------
Foundations module: what an embedding *is*, and three ways to produce one.

This file defines an `EmbeddingProvider` interface with three implementations:

1. SimpleHashEmbeddingProvider
   A deterministic, dependency-free, offline "embedding" built from character
   n-gram hashing. It has no real semantic understanding, but it is fast,
   reproducible, and requires no model download or API key. This is what
   the test suite uses, and it's a good way to *see* the mechanics of an
   embedding (a fixed-length vector) before worrying about model quality.

2. SentenceTransformerEmbeddingProvider
   A real, local, semantically-meaningful embedding model
   (e.g. "all-MiniLM-L6-v2") via the `sentence-transformers` library.
   Requires `pip install sentence-transformers` and a one-time model
   download, but no API key.

3. OpenAIEmbeddingProvider
   Calls OpenAI's embeddings API (e.g. "text-embedding-3-small").
   Requires `pip install openai` and an `OPENAI_API_KEY` environment
   variable.

Run this file directly for a walkthrough demo:
    python embedding_demo.py
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np


class EmbeddingProvider(ABC):
    """Common interface every embedding backend implements."""

    #: dimensionality of vectors this provider returns
    dimension: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: list of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), self.dimension), dtype float32.
        """
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper to embed a single string -> 1D vector."""
        return self.embed([text])[0]


class SimpleHashEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic, offline, dependency-free "embedding".

    Mechanism: for each text, generate character n-grams, hash each n-gram
    into a bucket in a fixed-size vector, and accumulate weighted counts.
    This is essentially a tiny, hashed bag-of-n-grams vector -- it captures
    surface-level lexical overlap (shared substrings), NOT meaning. Two
    sentences that mean the same thing but use different words will NOT
    score highly similar here. That's an intentional, useful limitation to
    see firsthand before moving to real semantic models.

    Good for: unit tests, offline demos, understanding vector mechanics.
    Bad for: any real semantic search or similarity task.
    """

    def __init__(self, dimension: int = 256, ngram_sizes: Sequence[int] = (3, 4, 5)):
        self.dimension = dimension
        self.ngram_sizes = ngram_sizes

    def _ngrams(self, text: str) -> List[str]:
        text = text.lower().strip()
        text = f" {text} "  # pad so edge n-grams are captured
        grams = []
        for n in self.ngram_sizes:
            grams.extend(text[i : i + n] for i in range(len(text) - n + 1))
        return grams

    def _hash_bucket(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for gram in self._ngrams(text):
                bucket = self._hash_bucket(gram)
                vectors[row, bucket] += 1.0
            # L2-normalize so cosine similarity behaves well
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Real local semantic embeddings via sentence-transformers.
    Requires: pip install sentence-transformers
    First call downloads the model weights (~80MB for MiniLM) and caches them.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.astype(np.float32)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Embeddings via the OpenAI API.
    Requires: pip install openai
    Requires: OPENAI_API_KEY environment variable set.
    """

    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is not installed. Run: pip install openai") from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY env var or pass api_key=."
            )

        self._client = OpenAI(api_key=key)
        self.model_name = model_name
        self.dimension = self._DIMENSIONS.get(model_name, 1536)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model_name, input=list(texts))
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return vectors


def get_default_provider() -> EmbeddingProvider:
    """
    Pick the best available provider without requiring configuration:
    OpenAI (if key set) > sentence-transformers (if installed) > hash fallback.
    """
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddingProvider()
        except Exception:
            pass
    try:
        return SentenceTransformerEmbeddingProvider()
    except ImportError:
        pass
    return SimpleHashEmbeddingProvider()


def _demo() -> None:
    from similarity import cosine_similarity

    provider = get_default_provider()
    print(f"Using provider: {provider.__class__.__name__} (dim={provider.dimension})\n")

    sentences = [
        "The cat sat on the mat.",
        "A cat was sitting on a mat.",
        "The stock market fell sharply today.",
        "Agentic AI systems can plan and use tools autonomously.",
    ]

    vectors = provider.embed(sentences)
    print(f"Embedded {len(sentences)} sentences into shape {vectors.shape}\n")

    print("Pairwise cosine similarity:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"  [{i}] vs [{j}]: {sim:.4f}")
            print(f"       '{sentences[i]}'")
            print(f"       '{sentences[j]}'")


if __name__ == "__main__":
    _demo()
