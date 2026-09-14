"""
retrieval/semantic_retriever.py
----------------------------------
Ties embedding + vector store together into the query API:

    query -> query embedding -> vector similarity -> ranking -> top-k chunks
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from retrieval_engineering.embeddings.embedder import Embedder
from retrieval_engineering.models.document import RetrievalResult
from retrieval_engineering.vector_store.local_store import LocalVectorStore

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """High-level retrieval API: embed the query, search the store, return ranked chunks."""

    def __init__(self, embedder: Embedder, vector_store: LocalVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        query_vector = self.embedder.embed_text(query)
        results = self.vector_store.search(query_vector, k=top_k, filters=filters)
        logger.info("Retrieved %d result(s) for query=%r (top_k=%d)", len(results), query, top_k)
        return results
