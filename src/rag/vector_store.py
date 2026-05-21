"""
src/rag/vector_store.py
-----------------------
Week 6 — RAG: FAISS vector index for policy document search.

WHAT THIS FILE DOES
-------------------
1. Takes PolicyChunk objects from document_loader.py.
2. Embeds each chunk's text into a 384-dim vector via embedder.py.
3. Stores the vectors in a FAISS index (Facebook AI Similarity Search).
4. At search time, embeds the query, finds the top-K closest vectors,
   and returns the matching PolicyChunk objects.

WHY FAISS?
----------
  - Runs 100% locally — no SaaS subscription (Pinecone costs $70+/month)
  - Millisecond search even at 10M+ vectors
  - Used in production at Meta, Microsoft, Google
  - Zero network calls — HIPAA safe

HOW IT WORKS
------------
FAISS stores vectors in a flat array and uses L2 (Euclidean) distance
to find the nearest neighbours. Since our embeddings are L2-normalized
(done in embedder.py), L2 distance is equivalent to cosine similarity:

    cosine_similarity(a, b) = 1 - (L2_distance(a, b)² / 2)

So searching by L2 distance on normalized vectors IS cosine similarity
search. This is a standard trick used in production vector search systems.

ARTIFACTS SAVED
---------------
  models/policy_index.faiss   — the FAISS index (binary file)
  models/policy_chunks.json   — chunk text + metadata (for retrieval)

The index and chunks are saved separately because FAISS stores only
vectors (numbers), not the text. When search returns "vector at index 7
is the closest match", we look up index 7 in policy_chunks.json to get
the actual text and source_file.

Usage:
    # Build index (run once after creating/updating policy documents)
    python -m src.rag.vector_store

    # Search from code
    from src.rag.vector_store import search
    results = search("high billing remediation", top_k=3)
"""

import json
import os
from pathlib import Path
from dataclasses import asdict

import faiss
import numpy as np

from src.core.logger import get_logger
from src.core.error_codes import ErrorCode
from src.core.exceptions import RAGError
from src.rag.document_loader import PolicyChunk, load_policy_documents
from src.rag.embedder import embed_texts, embed_query, EMBEDDING_DIM

# --------------------------------------------------------------------------- #
#  Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

# FAISS index file — binary format, not human-readable
INDEX_PATH  = MODELS_DIR / "policy_index.faiss"

# Chunk metadata file — JSON, stores the text and source info for each vector
CHUNKS_PATH = MODELS_DIR / "policy_chunks.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Module-level cache                                                           #
# --------------------------------------------------------------------------- #
# Loaded lazily on first search() call, then cached for the process lifetime.
_index:  faiss.IndexFlatL2 | None = None
_chunks: list[PolicyChunk] | None = None


# --------------------------------------------------------------------------- #
#  Build the index                                                              #
# --------------------------------------------------------------------------- #

def build_index(chunks: list[PolicyChunk] | None = None) -> int:
    """
    Build a FAISS index from policy chunks and save to disk.

    This function is called ONCE after creating or updating policy documents.
    It is NOT called during normal search operations — search() loads the
    pre-built index from disk.

    Steps:
      1. Load policy chunks (if not provided).
      2. Embed all chunk texts into 384-dim vectors.
      3. Add vectors to a FAISS IndexFlatL2 index.
      4. Save the index and chunk metadata to models/.

    Args:
        chunks: Pre-loaded PolicyChunk list (optional).
                If None, calls load_policy_documents() automatically.

    Returns:
        Number of chunks indexed.

    Raises:
        RAGError: If embedding or index saving fails.
    """
    logger.info("=== Building FAISS Policy Index ===")

    # Step 1: Load chunks if not provided
    if chunks is None:
        chunks = load_policy_documents()

    if len(chunks) == 0:
        logger.error(
            "[%s] No chunks to index — policy documents may be empty.",
            ErrorCode.RAG_NO_DOCUMENTS,
        )
        raise RAGError(
            error_code=ErrorCode.RAG_NO_DOCUMENTS,
            message="No chunks to index. Check data/policies/ for .txt files.",
        )

    # Step 2: Embed all chunk texts in one batch (efficient)
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)

    logger.info(
        "Chunks embedded. count=%d embedding_shape=%s",
        len(texts), embeddings.shape,
    )

    # Step 3: Build FAISS index
    # IndexFlatL2 = exact nearest-neighbour search using L2 distance.
    # For < 100K vectors this is fast enough. For millions, switch to
    # IndexIVFFlat (approximate search with inverted file index).
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)  # type: ignore[call-arg]  # SWIG binding — Pylance can't see real signature

    logger.info(
        "FAISS index built. vectors=%d dimension=%d",
        index.ntotal, EMBEDDING_DIM,
    )

    # Step 4: Save artifacts
    try:
        faiss.write_index(index, str(INDEX_PATH))
    except Exception as exc:
        logger.error(
            "[%s] Failed to save FAISS index. path=%s error=%s",
            ErrorCode.RAG_INDEX_NOT_FOUND, INDEX_PATH, str(exc),
        )
        raise RAGError(
            error_code=ErrorCode.RAG_INDEX_NOT_FOUND,
            message=f"Failed to save FAISS index: {exc}",
        ) from exc

    # Save chunk metadata as JSON so we can retrieve text after vector search
    try:
        chunks_data = [asdict(c) for c in chunks]
        with open(CHUNKS_PATH, "w", encoding="utf-8") as fp:
            json.dump(chunks_data, fp, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error(
            "[%s] Failed to save chunk metadata. path=%s error=%s",
            ErrorCode.RAG_INDEX_NOT_FOUND, CHUNKS_PATH, str(exc),
        )
        raise RAGError(
            error_code=ErrorCode.RAG_INDEX_NOT_FOUND,
            message=f"Failed to save chunk metadata: {exc}",
        ) from exc

    logger.info(
        "Index saved. index_path=%s chunks_path=%s",
        INDEX_PATH, CHUNKS_PATH,
    )
    logger.info("=== FAISS Policy Index Built ===")

    return index.ntotal


# --------------------------------------------------------------------------- #
#  Load the index (lazy, cached)                                                #
# --------------------------------------------------------------------------- #

def _load_index() -> None:
    """
    Load the FAISS index and chunk metadata from disk.

    Called automatically on the first search() call. Cached for subsequent
    calls. This function is internal — external code calls search() which
    handles lazy-loading.

    Raises:
        RAGError: If index or chunk files are missing or corrupt.
    """
    global _index, _chunks

    # Load FAISS index
    if not INDEX_PATH.exists():
        logger.error(
            "[%s] FAISS index not found. path=%s",
            ErrorCode.RAG_INDEX_NOT_FOUND, INDEX_PATH,
        )
        raise RAGError(
            error_code=ErrorCode.RAG_INDEX_NOT_FOUND,
            message=f"FAISS index not found: {INDEX_PATH}. "
                    f"Run: python -m src.rag.vector_store",
        )

    try:
        _index = faiss.read_index(str(INDEX_PATH))
    except Exception as exc:
        logger.error(
            "[%s] Failed to load FAISS index. error=%s",
            ErrorCode.RAG_INDEX_NOT_FOUND, str(exc),
        )
        raise RAGError(
            error_code=ErrorCode.RAG_INDEX_NOT_FOUND,
            message=f"Failed to load FAISS index: {exc}",
        ) from exc

    # Load chunk metadata
    if not CHUNKS_PATH.exists():
        logger.error(
            "[%s] Chunk metadata not found. path=%s",
            ErrorCode.RAG_INDEX_NOT_FOUND, CHUNKS_PATH,
        )
        raise RAGError(
            error_code=ErrorCode.RAG_INDEX_NOT_FOUND,
            message=f"Chunk metadata not found: {CHUNKS_PATH}",
        )

    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as fp:
            chunks_data = json.load(fp)
        _chunks = [PolicyChunk(**d) for d in chunks_data]
    except json.JSONDecodeError as exc:
        logger.error(
            "[%s] Chunk metadata is corrupt (invalid JSON). path=%s error=%s",
            ErrorCode.RAG_INDEX_NOT_FOUND, CHUNKS_PATH, str(exc),
        )
        raise RAGError(
            error_code=ErrorCode.RAG_INDEX_NOT_FOUND,
            message=f"policy_chunks.json is corrupt: {exc}",
        ) from exc
    except Exception as exc:
        logger.error(
            "[%s] Failed to load chunk metadata. error=%s",
            ErrorCode.RAG_INDEX_NOT_FOUND, str(exc),
        )
        raise RAGError(
            error_code=ErrorCode.RAG_INDEX_NOT_FOUND,
            message=f"Failed to load chunk metadata: {exc}",
        ) from exc

    logger.info(
        "FAISS index loaded. vectors=%d chunks=%d",
        _index.ntotal, len(_chunks),
    )


# --------------------------------------------------------------------------- #
#  Search                                                                       #
# --------------------------------------------------------------------------- #

def search(query: str, top_k: int = 3) -> list[tuple[PolicyChunk, float]]:
    """
    Search the FAISS index for the most relevant policy chunks.

    Args:
        query: Search string (e.g. "WARN_HIGH_BILLING",
               "missing diagnosis code remediation").
        top_k: Number of results to return (default 3).

    Returns:
        List of (PolicyChunk, distance) tuples, sorted by relevance
        (smallest distance = most relevant). Distance is L2 distance
        on normalized vectors — smaller = more similar.

    Raises:
        RAGError: If index is not built, or search fails.
    """
    # Lazy-load index on first call
    if _index is None or _chunks is None:
        _load_index()

    # Embed the query using the same model that embedded the chunks
    query_vector = embed_query(query)

    try:
        # FAISS search returns:
        #   distances: shape (1, top_k) — L2 distances
        #   indices:   shape (1, top_k) — integer positions in the index
        distances, indices = _index.search(query_vector, top_k)  # type: ignore[union-attr]
    except Exception as exc:
        logger.error(
            "[%s] FAISS search failed. query=%s error=%s",
            ErrorCode.RAG_EMPTY_RESULTS, query, str(exc),
        )
        raise RAGError(
            error_code=ErrorCode.RAG_EMPTY_RESULTS,
            message=f"FAISS search failed: {exc}",
        ) from exc

    # Build result list — map indices back to PolicyChunk objects
    results: list[tuple[PolicyChunk, float]] = []
    for i in range(top_k):
        idx = int(indices[0][i])
        dist = float(distances[0][i])

        # FAISS can return -1 for index if there are fewer vectors than top_k
        if idx < 0 or idx >= len(_chunks):  # type: ignore[arg-type]
            continue

        results.append((_chunks[idx], dist))  # type: ignore[index]

    if len(results) == 0:
        logger.warning(
            "[%s] Search returned 0 results. query=%s",
            ErrorCode.RAG_EMPTY_RESULTS, query,
        )

    logger.debug(
        "Search complete. query='%s' results=%d top_source=%s",
        query, len(results),
        results[0][0].source_file if results else "N/A",
    )

    return results


# --------------------------------------------------------------------------- #
#  Entry point — build the index                                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    is_production = os.getenv("APP_ENV", "development") == "production"

    try:
        count = build_index()
    except RAGError as exc:
        logger.error("Index build failed. code=%s error=%s", exc.error_code, exc.message)
        raise SystemExit(1) from exc

    if not is_production:
        print(f"\nFAISS Index Built:")
        print(f"  Vectors indexed : {count}")
        print(f"  Index file      : {INDEX_PATH}")
        print(f"  Chunks file     : {CHUNKS_PATH}")
