"""
tests/test_loader.py
----------------------
Unit tests for ingestion/loader.py. Uses tmp_path so these tests never
touch the real knowledge base and are safe to run in any order.
"""

from pathlib import Path

import pytest

from retrieval_engineering.ingestion.loader import discover_documents, load_document, load_documents


def _write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestDiscoverDocuments:
    def test_finds_md_and_txt_only(self, tmp_path: Path):
        _write(tmp_path, "a/one.md", "# One")
        _write(tmp_path, "a/two.txt", "two")
        _write(tmp_path, "a/ignore.json", "{}")
        _write(tmp_path, "a/ignore.py", "print(1)")

        found = discover_documents(tmp_path)
        names = sorted(p.name for p in found)
        assert names == ["one.md", "two.txt"]

    def test_missing_root_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            discover_documents(tmp_path / "does-not-exist")

    def test_is_recursive(self, tmp_path: Path):
        _write(tmp_path, "top.md", "top")
        _write(tmp_path, "nested/deep/leaf.md", "leaf")

        found = discover_documents(tmp_path)
        assert len(found) == 2


class TestLoadDocument:
    def test_preserves_source_filename_type_content(self, tmp_path: Path):
        path = _write(tmp_path, "aws/eks.md", "# EKS\n\nSome content.")

        doc = load_document(path, root=tmp_path)

        assert doc.source == "aws/eks.md"
        assert doc.filename == "eks.md"
        assert doc.document_type == "markdown"
        assert doc.content == "# EKS\n\nSome content."

    def test_txt_document_type(self, tmp_path: Path):
        path = _write(tmp_path, "notes.txt", "plain text")
        doc = load_document(path, root=tmp_path)
        assert doc.document_type == "text"

    def test_id_is_deterministic_for_same_source(self, tmp_path: Path):
        path = _write(tmp_path, "aws/eks.md", "# EKS")
        doc1 = load_document(path, root=tmp_path)
        doc2 = load_document(path, root=tmp_path)
        assert doc1.id == doc2.id

    def test_metadata_is_attached(self, tmp_path: Path):
        path = _write(tmp_path, "aws/eks.md", "# EKS\n\ncontent")
        doc = load_document(path, root=tmp_path)
        assert doc.metadata["category"] == "aws"
        assert doc.metadata["topic"] == "eks"


class TestLoadDocuments:
    def test_loads_all_supported_files(self, tmp_path: Path):
        _write(tmp_path, "aws/eks.md", "# EKS")
        _write(tmp_path, "aws/iam.md", "# IAM")
        _write(tmp_path, "notes.txt", "note")

        docs = load_documents(tmp_path)
        assert len(docs) == 3
        assert {d.source for d in docs} == {"aws/eks.md", "aws/iam.md", "notes.txt"}
