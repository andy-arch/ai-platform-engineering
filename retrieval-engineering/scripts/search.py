"""
scripts/search.py
--------------------
CLI: query -> query embedding -> vector similarity -> ranking -> top-k chunks.

Usage:
    python scripts/search.py "How does EKS authentication work?"
    python scripts/search.py "How does RBAC work?" --top-k 3 --category kubernetes
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval_engineering.embeddings.embedder import Embedder
from retrieval_engineering.retrieval.semantic_retriever import SemanticRetriever
from retrieval_engineering.vector_store.local_store import LocalVectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "index" / "kb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local vector index.")
    parser.add_argument("query", type=str)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", type=str, default=None, help="optional metadata filter")
    args = parser.parse_args()

    store = LocalVectorStore.load(args.index_path)
    retriever = SemanticRetriever(embedder=Embedder(), vector_store=store)

    filters = {"category": args.category} if args.category else None
    results = retriever.retrieve(args.query, top_k=args.top_k, filters=filters)

    print(f"Query:\n{args.query}\n")
    print("Top results:\n")
    for i, result in enumerate(results, start=1):
        preview = result.chunk.text[:200].replace("\n", " ")
        suffix = "..." if len(result.chunk.text) > 200 else ""
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   Source: {result.chunk.source}")
        print(f"   {preview}{suffix}\n")


if __name__ == "__main__":
    main()
