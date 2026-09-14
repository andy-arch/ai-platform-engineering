"""
ingestion/loader.py
--------------------
Discovers and loads documents (.md, .txt) from a knowledge base directory
into the internal Document representation.

This module is deliberately generic -- it knows nothing about "aws" or
"kubernetes" or any other topic. ingestion/metadata.py is what derives
topic/category from path structure, so this loader is reusable for any
directory of markdown/text documents.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

from retrieval_engineering.ingestion.metadata import derive_metadata
from retrieval_engineering.models.document import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md": "markdown", ".txt": "text"}


def discover_documents(root: str | Path) -> List[Path]:
    """Find every supported (.md, .txt) file under `root`, sorted for determinism."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Knowledge base root does not exist: {root}")

    paths = [
        p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    logger.info("Discovered %d document(s) under %s", len(paths), root)
    return paths


def _document_id(source: str) -> str:
    """Deterministic id derived from the document's relative path."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def load_document(path: Path, root: Path) -> Document:
    """Load a single file into a Document. `root` is used to compute `source`."""
    document_type = SUPPORTED_EXTENSIONS[path.suffix.lower()]
    content = path.read_text(encoding="utf-8")
    source = path.relative_to(root).as_posix()
    metadata = derive_metadata(source=source, document_type=document_type, content=content)

    return Document(
        id=_document_id(source),
        source=source,
        filename=path.name,
        document_type=document_type,
        content=content,
        metadata=metadata,
    )


def load_documents(root: str | Path) -> List[Document]:
    """Discover and load every supported document under `root`."""
    root = Path(root)
    paths = discover_documents(root)
    documents = [load_document(path, root) for path in paths]
    logger.info("Loaded %d document(s) from %s", len(documents), root)
    return documents
