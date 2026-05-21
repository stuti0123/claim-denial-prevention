"""
Unit Tests — Week 6: RAG Layer (document_loader, embedder, vector_store, retriever)

Test strategy:
  - document_loader: test chunking logic with synthetic text, test error on empty dir
  - embedder: test output shape, dtype, determinism (same input → same output)
  - vector_store: test build + search round-trip with real policy docs
  - retriever: test that each validation flag retrieves from the correct policy file

NOTE: Tests that use the embedding model (TestEmbedder, TestVectorStore,
TestRetriever) will download the model on first run (~80MB). Subsequent
runs use the cached model from ~/.cache/huggingface/.
"""

import json
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import RAGError
from src.rag.document_loader import (
    PolicyChunk,
    _chunk_text,
    load_policy_documents,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
)


# --------------------------------------------------------------------------- #
#  Tests: document_loader — _chunk_text                                         #
# --------------------------------------------------------------------------- #

class TestChunkText:
    """Test the internal _chunk_text() function with synthetic text."""

    def test_empty_text_returns_empty(self) -> None:
        """An empty string should produce zero chunks."""
        assert _chunk_text("") == []

    def test_short_text_single_chunk(self) -> None:
        """Text shorter than CHUNK_SIZE_WORDS produces exactly one chunk."""
        short = "hello world this is a test"
        chunks = _chunk_text(short, chunk_size=200, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == short

    def test_chunking_produces_multiple_chunks(self) -> None:
        """Text longer than chunk_size should be split into multiple chunks."""
        # Create text with exactly 400 words
        words = [f"word{i}" for i in range(400)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=200, overlap=50)
        # With 400 words, chunk_size=200, overlap=50, step=150:
        # Chunk 0: words 0-199, Chunk 1: words 150-349, Chunk 2: words 300-399
        assert len(chunks) == 3

    def test_overlap_shares_words(self) -> None:
        """Consecutive chunks must share `overlap` number of words at boundaries."""
        words = [f"w{i}" for i in range(300)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=200, overlap=50)
        # The last 50 words of chunk 0 should appear at the start of chunk 1
        chunk0_words = chunks[0].split()
        chunk1_words = chunks[1].split()
        overlap_from_0 = chunk0_words[-50:]
        overlap_from_1 = chunk1_words[:50]
        assert overlap_from_0 == overlap_from_1


# --------------------------------------------------------------------------- #
#  Tests: document_loader — load_policy_documents                               #
# --------------------------------------------------------------------------- #

class TestLoadPolicyDocuments:
    """Test loading policy documents from disk."""

    def test_raises_on_missing_directory(self, tmp_path: Path) -> None:
        """Should raise RAGError if the policies directory doesn't exist."""
        fake_dir = tmp_path / "nonexistent"
        with pytest.raises(RAGError) as exc_info:
            load_policy_documents(policies_dir=fake_dir)
        assert exc_info.value.error_code == "RAG-5001"

    def test_raises_on_empty_directory(self, tmp_path: Path) -> None:
        """Should raise RAGError if the policies directory has no .txt files."""
        empty_dir = tmp_path / "empty_policies"
        empty_dir.mkdir()
        with pytest.raises(RAGError) as exc_info:
            load_policy_documents(policies_dir=empty_dir)
        assert exc_info.value.error_code == "RAG-5001"

    def test_loads_txt_files(self, tmp_path: Path) -> None:
        """Should load .txt files and produce PolicyChunk objects."""
        # Create a small policy file
        policy_dir = tmp_path / "policies"
        policy_dir.mkdir()
        (policy_dir / "test_policy.txt").write_text("This is a test policy document.")

        chunks = load_policy_documents(policies_dir=policy_dir)
        assert len(chunks) >= 1
        assert isinstance(chunks[0], PolicyChunk)
        assert chunks[0].source_file == "test_policy.txt"

    def test_chunk_id_format(self, tmp_path: Path) -> None:
        """chunk_id should follow the format '{filename_stem}_{index}'."""
        policy_dir = tmp_path / "policies"
        policy_dir.mkdir()
        (policy_dir / "billing.txt").write_text("Short text.")

        chunks = load_policy_documents(policies_dir=policy_dir)
        assert chunks[0].chunk_id == "billing_0"


# --------------------------------------------------------------------------- #
#  Tests: embedder                                                              #
# --------------------------------------------------------------------------- #

class TestEmbedder:
    """Test the sentence-transformers embedding module."""

    def test_embed_shape(self) -> None:
        """embed_texts should return shape (n, 384) for n input texts."""
        from src.rag.embedder import embed_texts, EMBEDDING_DIM

        texts = ["hello world", "test document"]
        vectors = embed_texts(texts)
        assert vectors.shape == (2, EMBEDDING_DIM)

    def test_embed_dtype_float32(self) -> None:
        """Embeddings must be float32 (FAISS requirement)."""
        from src.rag.embedder import embed_texts

        vectors = embed_texts(["test"])
        assert vectors.dtype == np.float32

    def test_embed_deterministic(self) -> None:
        """Same input text must produce the same embedding vector every time."""
        from src.rag.embedder import embed_texts

        v1 = embed_texts(["high billing remediation"])
        v2 = embed_texts(["high billing remediation"])
        np.testing.assert_array_almost_equal(v1, v2, decimal=5)

    def test_embed_empty_list(self) -> None:
        """embed_texts([]) should return an empty array with correct shape."""
        from src.rag.embedder import embed_texts, EMBEDDING_DIM

        result = embed_texts([])
        assert result.shape == (0, EMBEDDING_DIM)

    def test_embed_query_shape(self) -> None:
        """embed_query should return shape (1, 384)."""
        from src.rag.embedder import embed_query, EMBEDDING_DIM

        vec = embed_query("test query")
        assert vec.shape == (1, EMBEDDING_DIM)


# --------------------------------------------------------------------------- #
#  Tests: vector_store — build + search                                         #
# --------------------------------------------------------------------------- #

class TestVectorStore:
    """Test FAISS index build and search with real policy documents."""

    @pytest.fixture(autouse=True)
    def _build_index(self) -> None:
        """Build the FAISS index before running search tests.

        Uses the real policy documents from data/policies/.
        This fixture runs once per test method (autouse=True).
        """
        from src.rag.vector_store import build_index
        build_index()

    def test_index_file_created(self) -> None:
        """build_index() must create the FAISS index file on disk."""
        from src.rag.vector_store import INDEX_PATH
        assert INDEX_PATH.exists()

    def test_chunks_file_created(self) -> None:
        """build_index() must create the chunk metadata JSON file."""
        from src.rag.vector_store import CHUNKS_PATH
        assert CHUNKS_PATH.exists()

    def test_search_returns_results(self) -> None:
        """search() should return a non-empty list for a valid query."""
        from src.rag.vector_store import search
        results = search("high billing remediation", top_k=3)
        assert len(results) > 0
        assert isinstance(results[0][0], PolicyChunk)

    def test_search_top_k_respected(self) -> None:
        """search(top_k=2) should return at most 2 results."""
        from src.rag.vector_store import search
        results = search("missing diagnosis code", top_k=2)
        assert len(results) <= 2

    def test_search_returns_distance(self) -> None:
        """Each search result must include a float distance score."""
        from src.rag.vector_store import search
        results = search("incomplete claim", top_k=1)
        assert len(results) == 1
        chunk, distance = results[0]
        assert isinstance(distance, float)
        assert distance >= 0.0


# --------------------------------------------------------------------------- #
#  Tests: retriever                                                             #
# --------------------------------------------------------------------------- #

class TestRetriever:
    """Test the high-level retrieval API."""

    @pytest.fixture(autouse=True)
    def _build_index(self) -> None:
        """Ensure FAISS index exists before retriever tests."""
        from src.rag.vector_store import build_index
        build_index()

    def test_high_billing_returns_billing_doc(self) -> None:
        """WARN_HIGH_BILLING should retrieve from billing_guidelines.txt."""
        from src.rag.retriever import retrieve
        result = retrieve("WARN_HIGH_BILLING")
        assert result.found is True
        assert "billing" in result.source_file.lower()

    def test_incomplete_claim_returns_incomplete_doc(self) -> None:
        """ERR_INCOMPLETE_CLAIM should retrieve from incomplete_claim_policy.txt."""
        from src.rag.retriever import retrieve
        result = retrieve("ERR_INCOMPLETE_CLAIM")
        assert result.found is True
        assert "incomplete" in result.source_file.lower()

    def test_missing_diagnosis_returns_diagnosis_doc(self) -> None:
        """WARN_MISSING_DIAGNOSIS should retrieve from diagnosis_code_policy.txt."""
        from src.rag.retriever import retrieve
        result = retrieve("WARN_MISSING_DIAGNOSIS")
        assert result.found is True
        assert "diagnosis" in result.source_file.lower()

    def test_result_has_top_text(self) -> None:
        """RetrievalResult.top_text should be a non-empty string."""
        from src.rag.retriever import retrieve
        result = retrieve("WARN_MISSING_PROCEDURE")
        assert result.found is True
        assert len(result.top_text) > 0

    def test_retrieve_multiple(self) -> None:
        """retrieve_multiple should return results for all flags."""
        from src.rag.retriever import retrieve_multiple
        flags = ["WARN_HIGH_BILLING", "ERR_INCOMPLETE_CLAIM"]
        results = retrieve_multiple(flags)
        assert len(results) == 2
        assert all(r.found for r in results.values())

    def test_exception_hierarchy(self) -> None:
        """RAGError must be a subclass of ClaimDenialSystemError."""
        from src.core.exceptions import ClaimDenialSystemError
        err = RAGError(error_code="RAG-5001", message="test")
        assert isinstance(err, ClaimDenialSystemError)
        assert err.error_code == "RAG-5001"
