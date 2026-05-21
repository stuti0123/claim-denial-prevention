"""
src/rag/embedder.py
-------------------
Week 6 — RAG: Convert text to numeric vectors using a local embedding model.
-------------------
Takes a list of text strings (policy chunks or queries) and returns their
numeric vector representations (embeddings). These embeddings are what
FAISS stores and searches against.

Think of an embedding as a "meaning fingerprint":
  - "billed amount exceeds 3 times expected cost" → [0.12, -0.44, 0.08, ...]
  - "high billing flag remediation"                → [0.11, -0.42, 0.09, ...]
  - "missing diagnosis code"                       → [-0.33, 0.18, 0.55, ...]

The first two texts are about the same topic (billing), so their vectors are
close together in 384-dimensional space. The third text is about something
different (diagnosis codes), so its vector is far away. FAISS finds the
closest vectors — that's how retrieval works.

MODEL CHOICE: all-MiniLM-L6-v2
-------------------------------
  - Runs 100% locally on CPU — NO data sent to any external API
  - 384-dimensional vectors (small, fast)
  - Downloads once (~80MB) to ~/.cache/huggingface/ and reuses from cache
  - No API key, no cost per query, no rate limits
  - Suitable for production healthcare systems under HIPAA

HIPAA COMPLIANCE
----------------
Must NOT send protected health information (PHI) to external
APIs without a signed Business Associate Agreement (BAA). By using a local
embedding model, we eliminate this risk entirely. The policy documents
themselves contain no PHI (they are generic billing rules), and the claim
data never passes through this module — only the flag names are used as
queries (e.g. "WARN_HIGH_BILLING"), not actual patient data.

Usage:
    from src.rag.embedder import embed_texts, embed_query
    vectors = embed_texts(["policy text chunk 1", "policy text chunk 2"])
    query_vec = embed_query("WARN_HIGH_BILLING remediation")
"""

from pathlib import Path
from typing import Optional

import numpy as np

from src.core.logger import get_logger
from src.core.error_codes import ErrorCode
from src.core.exceptions import RAGError

# --------------------------------------------------------------------------- #
#  Configuration                                                                #
# --------------------------------------------------------------------------- #

# Model name — downloaded from HuggingFace Hub on first use, cached locally.
# all-MiniLM-L6-v2 produces 384-dimensional float32 vectors.
# If you want higher quality (but slower), use "all-mpnet-base-v2" (768-dim).
MODEL_NAME: str = "all-MiniLM-L6-v2"

# Embedding dimension — must match the model output size.
# Used by vector_store.py to initialize the FAISS index.
EMBEDDING_DIM: int = 384

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Model cache — loaded once per process                                        #
# --------------------------------------------------------------------------- #

# The model is loaded lazily on first call and cached for subsequent calls.
# This avoids the 2-3 second load time on every function call.
_model: Optional[object] = None


def _load_model() -> None:
    """
    Load the sentence-transformers model into memory.

    Called automatically on the first call to embed_texts() or embed_query().
    Subsequent calls use the cached _model object.

    WHY lazy loading?
    Loading a transformer model takes 2-3 seconds. If we loaded it at
    import time, every module that imports embedder.py would pay that cost
    even if they never call embed_texts(). Lazy loading means the cost is
    only paid when actually needed.

    Raises:
        RAGError: If the model fails to load (network error on first download,
                  corrupt cache, incompatible Python version).
    """
    global _model

    try:
        # Import here (not at module level) to avoid paying the import cost
        # when this module is imported but not used.
        from sentence_transformers import SentenceTransformer
        import os

        # Check for local ModelScope cache path (bypass corporate network block on HuggingFace)
        model_path = os.path.expanduser("~/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2")
        target_model = model_path if os.path.exists(model_path) else MODEL_NAME

        _model = SentenceTransformer(target_model)
        logger.info(
            "Embedding model loaded. model=%s dim=%d path_resolved=%s",
            MODEL_NAME, EMBEDDING_DIM, target_model,
        )

    except Exception as exc:
        logger.error(
            "[%s] Failed to load embedding model. model=%s error=%s",
            ErrorCode.RAG_EMBED_FAILED, MODEL_NAME, str(exc),
        )
        raise RAGError(
            error_code=ErrorCode.RAG_EMBED_FAILED,
            message=f"Failed to load embedding model '{MODEL_NAME}': {exc}",
        ) from exc


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

def get_model() -> object:
    """
    Return the loaded embedding model, loading it if necessary.

    WHY a public function instead of importing _model directly?
    Same pattern as predictor.get_model() — avoids hidden state coupling.
    External modules call get_model() which handles lazy-loading.

    Returns:
        The loaded SentenceTransformer model object.

    Raises:
        RAGError: If the model cannot be loaded.
    """
    if _model is None:
        _load_model()
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors.

    Used during index building to embed all policy chunks at once.
    Batch processing is much faster than embedding one text at a time
    because the model can process multiple texts in parallel on the CPU.

    Args:
        texts: List of text strings to embed. Each string is typically
               a PolicyChunk.text (~200 words).

    Returns:
        NumPy array of shape (len(texts), 384), dtype float32.
        Each row is the embedding vector for the corresponding text.

    Raises:
        RAGError: If embedding fails (model not loaded, OOM, etc.).
    """
    if _model is None:
        _load_model()

    if len(texts) == 0:
        logger.warning("embed_texts called with empty list — returning empty array.")
        return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

    try:
        # SentenceTransformer.encode() handles tokenization, padding,
        # and mean pooling internally. Returns np.ndarray float32.
        embeddings: np.ndarray = _model.encode(   # type: ignore[union-attr]
            texts,
            show_progress_bar=False,   # suppress tqdm bar — we use our logger
            convert_to_numpy=True,     # return np.ndarray, not torch.Tensor
            normalize_embeddings=True, # L2-normalize for cosine similarity in FAISS
        )

        logger.debug(
            "Texts embedded. count=%d shape=%s dtype=%s",
            len(texts), embeddings.shape, embeddings.dtype,
        )
        return embeddings

    except Exception as exc:
        logger.error(
            "[%s] Embedding failed. count=%d error=%s",
            ErrorCode.RAG_EMBED_FAILED, len(texts), str(exc),
        )
        raise RAGError(
            error_code=ErrorCode.RAG_EMBED_FAILED,
            message=f"Failed to embed {len(texts)} texts: {exc}",
        ) from exc


def embed_query(query: str) -> np.ndarray:
    """
    Convert a single query string into an embedding vector.

    Used at search time to embed the user's query before searching FAISS.

    WHY a separate function instead of embed_texts([query])?
    Semantic clarity. The caller reads `embed_query("high billing")`
    and immediately understands this is a single-query operation, not
    a batch operation. The internal implementation is the same.

    Args:
        query: Search query string (e.g. "WARN_HIGH_BILLING",
               "high billing remediation steps").

    Returns:
        NumPy array of shape (1, 384), dtype float32.

    Raises:
        RAGError: If embedding fails.
    """
    return embed_texts([query])
