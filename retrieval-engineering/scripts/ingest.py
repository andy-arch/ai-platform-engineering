"""
scripts/ingest.py
-------------------
CLI: discover -> load -> chunk -> embed -> store -> persist.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --chunk-size 300 --overlap 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from retrieval_engineering.embeddings.embedder import Embedder
from retrieval_engineering.ingestion.chunker import FixedSizeChunker, chunk_documents
from retrieval_engineering.ingestion.loader import load_documents
from retrieval_engineering.vector_store.local_store import LocalVectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_DIR = REPO_ROOT / "data" / "knowledge"
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "index" / "kb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a knowledge base into a local vector index.")
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    documents = load_documents(args.knowledge_dir)
    logger.info("Loaded %d document(s)", len(documents))

    chunker = FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap)
    chunks = chunk_documents(documents, strategy=chunker)
    logger.info("Produced %d chunk(s)", len(chunks))

    embedder = Embedder()
    logger.info(
        "Using embedding provider: %s (dim=%d)", embedder.provider.__class__.__name__, embedder.dimension
    )
    vectors = embedder.embed_documents([c.text for c in chunks])

    store = LocalVectorStore()
    store.add(chunks, vectors)
    store.save(args.index_path)

    print(f"Ingested {len(documents)} document(s) -> {len(chunks)} chunk(s).")
    print(f"Index saved to: {args.index_path}.vectors.npy / .payload.json")


if __name__ == "__main__":
    main()
