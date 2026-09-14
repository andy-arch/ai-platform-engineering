"""
tests/test_metadata.py
------------------------
Unit tests for ingestion/metadata.py -- deriving category/topic/title
from a document's path and content rather than hand-specifying it.
"""

from retrieval_engineering.ingestion.metadata import derive_metadata


class TestDeriveMetadata:
    def test_category_and_topic_from_path(self):
        meta = derive_metadata(source="aws/eks.md", document_type="markdown", content="# EKS")
        assert meta["category"] == "aws"
        assert meta["topic"] == "eks"
        assert meta["document_type"] == "markdown"
        assert meta["source"] == "aws/eks.md"

    def test_root_level_file_is_uncategorized(self):
        meta = derive_metadata(source="notes.md", document_type="markdown", content="# Notes")
        assert meta["category"] == "uncategorized"
        assert meta["topic"] == "notes"

    def test_nested_path_uses_top_level_directory_as_category(self):
        meta = derive_metadata(
            source="api-gateway/auth/jwt.md", document_type="markdown", content="# JWT"
        )
        assert meta["category"] == "api-gateway"
        assert meta["topic"] == "jwt"

    def test_title_extracted_from_h1(self):
        meta = derive_metadata(
            source="aws/eks.md", document_type="markdown", content="# Amazon EKS\n\nSome body text."
        )
        assert meta["title"] == "Amazon EKS"

    def test_no_title_when_no_h1_present(self):
        meta = derive_metadata(source="aws/eks.md", document_type="markdown", content="no heading here")
        assert "title" not in meta

    def test_text_documents_have_no_title_extraction(self):
        meta = derive_metadata(source="notes.txt", document_type="text", content="# looks like a heading")
        assert "title" not in meta
