"""
src/rag/document_loader.py
--------------------------
Week 6 — RAG: Load and chunk policy documents for vector embedding.

WHAT THIS FILE DOES
-------------------
1. Scans data/policies/ for .txt files.
2. Reads each file.
3. Splits the text into overlapping chunks of ~200 words.
4. Returns a list of PolicyChunk dataclasses, each carrying:
     - chunk_id:    unique identifier (filename + chunk index)
     - source_file: which policy document it came from
     - text:        the actual chunk content
     - metadata:    {source_file, chunk_index}

WHY CHUNKING?
-------------
Embedding models (like all-MiniLM-L6-v2) work on short text segments.
If you feed a full 500-word document as one vector, the model averages
out the meaning and loses specificity. A 200-word chunk about
"WARN_HIGH_BILLING remediation steps" will match the query
"high billing" much more precisely than the entire billing_guidelines.txt.

WHY OVERLAPPING?
----------------
Without overlap, a sentence at the boundary between two chunks gets
split in half — the retriever would never find it. A 50-word overlap
ensures boundary sentences appear in both the preceding and following
chunks, so no information is lost.

  - Every I/O operation wrapped in try/except
  - All errors use RAG-5xxx error codes from error_codes.py
  - Raises RAGError (from exceptions.py) — never bare Exception

HIPAA note: Policy documents contain NO patient data. They are
generic billing rules. Loading them is safe under all compliance
frameworks.

Usage:
    from src.rag.document_loader import load_policy_documents
    chunks = load_policy_documents()
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.core.logger import get_logger
from src.core.error_codes import ErrorCode
from src.core.exceptions import RAGError

# --------------------------------------------------------------------------- #
#  Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICIES_DIR = PROJECT_ROOT / "data" / "policies"

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Constants                                                                    #
# --------------------------------------------------------------------------- #

# Target chunk size in words. 200 words ≈ 1–2 paragraphs.
# This is small enough for the embedding model to capture specific meaning,
# but large enough to retain context about what policy section we're in.
CHUNK_SIZE_WORDS: int = 200

# Overlap between consecutive chunks in words.
# 50 words ≈ 2–3 sentences of shared context at boundaries.
CHUNK_OVERLAP_WORDS: int = 50


# --------------------------------------------------------------------------- #
#  Data structure                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class PolicyChunk:
    """
    A single text chunk extracted from a policy document.

    WHY a dataclass?
    ----------------
    Same pattern as PredictionResult and FeatureAttribution from Week 5.
    Typed output makes the contract explicit — the vector_store and retriever
    know exactly what fields to expect. No "dict key guessing."

    Attributes:
        chunk_id:    Unique identifier: "{filename}_{chunk_index}"
                     Used as a primary key when storing/retrieving chunks.
        source_file: Original filename (e.g. "billing_guidelines.txt").
                     Shown to the billing analyst so they know which policy
                     document the information came from.
        text:        The actual text content of this chunk (~200 words).
        metadata:    Additional key-value pairs for filtering or display.
    """
    chunk_id:    str
    source_file: str
    text:        str
    metadata:    dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Chunking logic                                                               #
# --------------------------------------------------------------------------- #

def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split a text string into overlapping word-based chunks.

    Algorithm:
      1. Split text into a list of words.
      2. Slide a window of `chunk_size` words across the list.
      3. Each window advances by (chunk_size - overlap) words.
      4. The last window may be smaller than chunk_size — that's fine.

    WHY word-based instead of character-based?
    Word-based chunks produce more natural text boundaries. Character-based
    chunking can split words in half (e.g. "reimburse|ment"), which hurts
    embedding quality.

    Args:
        text:       Raw text string to chunk.
        chunk_size: Target number of words per chunk.
        overlap:    Number of overlapping words between consecutive chunks.

    Returns:
        List of text chunks (strings). Empty list if text is empty.
    """
    words = text.split()
    if len(words) == 0:
        return []

    chunks: list[str] = []
    # Step size = chunk_size - overlap. E.g. 200 - 50 = 150 words per step.
    step = max(chunk_size - overlap, 1)

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # Stop if we've reached the end of the document
        if end >= len(words):
            break

    return chunks


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

def load_policy_documents(
    policies_dir: Path | None = None,
) -> list[PolicyChunk]:
    """
    Load all .txt policy documents and return chunked PolicyChunk objects.

    Steps:
      1. Scan the policies directory for .txt files.
      2. Read each file.
      3. Split into overlapping chunks.
      4. Wrap each chunk in a PolicyChunk dataclass.

    Args:
        policies_dir: Path to the policies directory.
                      Defaults to data/policies/ in the project root.

    Returns:
        List of PolicyChunk objects, sorted by (source_file, chunk_index).

    Raises:
        RAGError: If the policies directory is empty or does not exist.
    """
    if policies_dir is None:
        policies_dir = POLICIES_DIR

    # Check that the directory exists and is not empty
    if not policies_dir.exists():
        logger.error(
            "[%s] Policies directory not found. path=%s",
            ErrorCode.RAG_NO_DOCUMENTS, policies_dir,
        )
        raise RAGError(
            error_code=ErrorCode.RAG_NO_DOCUMENTS,
            message=f"Policies directory not found: {policies_dir}",
        )

    # Find all .txt files (sorted for deterministic chunk_id ordering)
    txt_files = sorted(policies_dir.glob("*.txt"))
    if len(txt_files) == 0:
        logger.error(
            "[%s] Policies directory is empty — no .txt files. path=%s",
            ErrorCode.RAG_NO_DOCUMENTS, policies_dir,
        )
        raise RAGError(
            error_code=ErrorCode.RAG_NO_DOCUMENTS,
            message=f"No .txt files found in {policies_dir}",
        )

    logger.info(
        "Found policy documents. count=%d dir=%s",
        len(txt_files), policies_dir,
    )

    all_chunks: list[PolicyChunk] = []

    for file_path in txt_files:
        # I/O boundary: wrap file read in try/except
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error(
                "[%s] Failed to read policy file. file=%s error=%s",
                ErrorCode.RAG_NO_DOCUMENTS, file_path.name, str(exc),
            )
            raise RAGError(
                error_code=ErrorCode.RAG_NO_DOCUMENTS,
                message=f"Failed to read policy file '{file_path.name}': {exc}",
            ) from exc

        # Chunk the document text
        text_chunks = _chunk_text(raw_text)

        for idx, chunk_text in enumerate(text_chunks):
            chunk = PolicyChunk(
                chunk_id=f"{file_path.stem}_{idx}",
                source_file=file_path.name,
                text=chunk_text,
                metadata={
                    "source_file": file_path.name,
                    "chunk_index": idx,
                    "total_chunks": len(text_chunks),
                },
            )
            all_chunks.append(chunk)

        logger.info(
            "Policy chunked. file=%s chunks=%d",
            file_path.name, len(text_chunks),
        )

    logger.info(
        "All policies loaded. total_chunks=%d total_files=%d",
        len(all_chunks), len(txt_files),
    )
    return all_chunks
