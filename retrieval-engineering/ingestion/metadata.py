"""
ingestion/metadata.py
----------------------
Derives chunk/document metadata from a document's path and structure,
rather than requiring it to be hand-specified per file. Given a source
path like "aws/eks.md" (relative to the knowledge root), this extracts:

    category = "aws"          # top-level directory
    topic    = "eks"          # filename without extension
    document_type = "markdown"

It also pulls the first Markdown H1 heading (if present) as a
human-readable `title`, since filenames like "eks.md" are terse.

This is deliberately simple, pattern-based derivation -- not a general
metadata-extraction framework. Its output shape is what later enables
metadata filtering in the vector store (e.g. "only search category=aws").
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Dict

_H1_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def derive_metadata(source: str, document_type: str, content: str) -> Dict[str, Any]:
    """
    Args:
        source: path relative to the knowledge root, using "/" separators
            (e.g. "aws/eks.md", or "notes.md" for a root-level file).
        document_type: "markdown" or "text".
        content: the document's raw text, used to pull a title if present.
    """
    path = PurePosixPath(source)
    parts = path.parts

    category = parts[0] if len(parts) > 1 else "uncategorized"
    topic = path.stem

    metadata: Dict[str, Any] = {
        "source": source,
        "category": category,
        "topic": topic,
        "document_type": document_type,
    }

    if document_type == "markdown":
        match = _H1_PATTERN.search(content)
        if match:
            metadata["title"] = match.group(1).strip()

    return metadata
