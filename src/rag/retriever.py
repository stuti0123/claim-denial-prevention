"""
src/rag/retriever.py
--------------------
Week 6 — RAG: High-level retrieval API for the Agent and Dashboard.

WHAT THIS FILE DOES
-------------------
This is the PUBLIC API for the RAG system. The Agent (Week 7) and
Dashboard (Week 8) call exactly ONE function:

    result = retrieve("WARN_HIGH_BILLING")

It returns a RetrievalResult dataclass containing:
  - The query used
  - The top matching policy chunks (with text and source file)
  - A formatted explanation string ready for display

This module sits on TOP of the RAG stack:
    retriever.py → vector_store.py → embedder.py → document_loader.py

The Agent does not need to know about FAISS, embeddings, or chunking.
It just calls retrieve() and gets a structured answer.

WHY A SEPARATE RETRIEVER?
--------------------------
  - vector_store.py handles the HOW (FAISS search mechanics)
  - retriever.py handles the WHAT (translating flag names to search queries,
    formatting results for the billing analyst)

This means if we switch from FAISS to Pinecone or Weaviate in the future,
only vector_store.py changes. retriever.py stays the same.

Usage:
    from src.rag.retriever import retrieve

    result = retrieve("WARN_HIGH_BILLING")
    print(result.top_text)
    # "Section 3: When a claim is flagged with WARN_HIGH_BILLING..."

    for chunk, score in result.chunks_with_scores:
        print(f"  {chunk.source_file}: {score:.4f}")
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.core.logger import get_logger
from src.core.error_codes import ErrorCode
from src.core.exceptions import RAGError
from src.rag.document_loader import PolicyChunk
from src.rag.vector_store import search

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Query expansion — maps validation flags to richer search queries             #
# --------------------------------------------------------------------------- #

# WHY QUERY EXPANSION?
# A raw flag name like "WARN_HIGH_BILLING" is very short. The embedding model
# works better with natural language. So we expand the flag name into a
# descriptive query that is more likely to match the relevant policy chunk.
#
# Example:
#   "WARN_HIGH_BILLING" → "high billing flag billed amount exceeds expected cost"
#   This richer query matches better against the billing_guidelines.txt chunks.

FLAG_QUERIES: dict[str, str] = {
    "WARN_HIGH_BILLING": (
        "high billing flag billed amount exceeds expected cost "
        "overbilling threshold remediation itemized receipts"
    ),
    "ERR_INCOMPLETE_CLAIM": (
        "incomplete claim both diagnosis and procedure code missing "
        "required fields automatic denial most severe validation error"
    ),
    "WARN_MISSING_DIAGNOSIS": (
        "missing diagnosis code ICD-10 clinical documentation "
        "medical necessity remediation steps"
    ),
    "WARN_MISSING_PROCEDURE": (
        "missing procedure code CPT code operative report "
        "service performed reimbursement"
    ),
    "WARN_MISSING_AMOUNT": (
        "missing billed amount charge capture financial requirement "
        "fee schedule remediation"
    ),
    "WARN_INVALID_DIAGNOSIS": (
        "invalid diagnosis code not in reference table typographical error "
        "outdated ICD-10 code correction"
    ),
}

# Default query for flags not in the expansion map — uses the raw flag name
DEFAULT_QUERY_PREFIX: str = "healthcare claim validation rule remediation "


# --------------------------------------------------------------------------- #
#  Return type                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class RetrievalResult:
    """
    Structured output from a policy retrieval operation.

    WHY a dataclass?
    Same pattern as PredictionResult (Week 5) and FeatureAttribution.
    Typed output = explicit contract = easy to test = easy to consume
    in the API and Dashboard.

    Attributes:
        query:              The expanded search query that was used.
        flag:               The original validation flag name.
        chunks_with_scores: List of (PolicyChunk, distance) tuples.
                            Sorted by relevance (smallest distance = best match).
        top_text:           Text from the best matching chunk (convenience field).
        source_file:        Source file of the best match (convenience field).
        found:              True if at least one relevant chunk was found.
    """
    query:              str
    flag:               str
    chunks_with_scores: list[tuple[PolicyChunk, float]] = field(default_factory=list)
    top_text:           str = ""
    source_file:        str = ""
    found:              bool = False


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

def retrieve(flag: str, top_k: int = 3) -> RetrievalResult:
    """
    Retrieve the most relevant policy text for a given validation flag.

    This is the MAIN ENTRY POINT for the RAG system. The Agent (Week 7)
    and Dashboard (Week 8) call this function with a flag name and get back
    the matching policy text.

    Steps:
      1. Expand the flag name into a richer search query.
      2. Search the FAISS index for the top-K matching chunks.
      3. Return a structured RetrievalResult.

    Args:
        flag:  Validation flag name (e.g. "WARN_HIGH_BILLING",
               "ERR_INCOMPLETE_CLAIM"). These are the same flag names
               used in validator.py and feature_engineer.py.
        top_k: Number of results to return (default 3).

    Returns:
        RetrievalResult with the matching policy chunks and text.

    Raises:
        RAGError: If the FAISS index is not built, or search fails.

    Example:
        result = retrieve("WARN_HIGH_BILLING")
        if result.found:
            print(f"Policy: {result.source_file}")
            print(f"Text: {result.top_text[:200]}...")
    """
    # Step 1: Expand the flag name into a descriptive query.
    # If the flag is in our expansion map, use the richer query.
    # Otherwise, prepend a general prefix to the raw flag name.
    query = FLAG_QUERIES.get(flag, DEFAULT_QUERY_PREFIX + flag)

    logger.info(
        "Retrieving policy for flag. flag=%s query_len=%d top_k=%d",
        flag, len(query), top_k,
    )

    # Step 2: Search the FAISS index
    try:
        search_results = search(query, top_k=top_k)
    except RAGError:
        # Already logged in vector_store.search() — propagate
        raise

    # Step 3: Build the RetrievalResult
    if len(search_results) == 0:
        logger.warning(
            "[%s] No policy chunks found for flag. flag=%s",
            ErrorCode.RAG_EMPTY_RESULTS, flag,
        )
        return RetrievalResult(
            query=query,
            flag=flag,
            found=False,
        )

    # Best match is the first result (lowest L2 distance = highest similarity)
    best_chunk, best_score = search_results[0]

    result = RetrievalResult(
        query=query,
        flag=flag,
        chunks_with_scores=search_results,
        top_text=best_chunk.text,
        source_file=best_chunk.source_file,
        found=True,
    )

    logger.info(
        "Policy retrieved. flag=%s source=%s score=%.4f chunks=%d",
        flag, best_chunk.source_file, best_score, len(search_results),
    )

    return result


def retrieve_multiple(flags: list[str], top_k: int = 3) -> dict[str, RetrievalResult]:
    """
    Retrieve policies for multiple validation flags at once.

    Used when a single claim has multiple flags (e.g. WARN_HIGH_BILLING
    AND WARN_MISSING_DIAGNOSIS). Returns one RetrievalResult per flag.

    Args:
        flags: List of validation flag names.
        top_k: Number of results per flag (default 3).

    Returns:
        Dict mapping flag name to its RetrievalResult.

    Raises:
        RAGError: If the FAISS index is not built, or search fails.
    """
    results: dict[str, RetrievalResult] = {}

    for flag in flags:
        results[flag] = retrieve(flag, top_k=top_k)

    logger.info(
        "Multi-flag retrieval complete. flags=%d found=%d",
        len(flags), sum(1 for r in results.values() if r.found),
    )
    return results
