"""
src/core/exceptions.py
-----------------------
Custom exception hierarchy for the Claim Denial System.

Three reasons custom exceptions matter:

1. SPECIFIC CATCHING: Callers can do `except IngestionError` instead of
   `except Exception`, which is too broad and hides bugs.

2. STRUCTURED CONTEXT: Each exception carries an error_code field, so the
   log line and the exception message always agree on the code.

3. ENFORCEMENT: If a new developer writes `raise ValueError("something broke")`
   instead of `raise SilverPipelineError(ErrorCode.SLV_HIGH_NULL_RATE, ...)`,
   the code review catches it. The hierarchy makes the right way obvious.

HIERARCHY
---------
    ClaimDenialSystemError          ← base for all system errors
    ├── IngestionError              ← Bronze layer (ING-1xxx)
    ├── SilverPipelineError         ← Silver cleaning/validation (SLV-2xxx)
    ├── GoldPipelineError           ← Gold feature engineering (GLD-3xxx)
    ├── ModelNotFoundError          ← ML artifacts missing (ML-4xxx)
    ├── PredictionError             ← ML inference failure (ML-4xxx)
    ├── RAGError                    ← RAG system failure (RAG-5xxx)
    ├── AgentError                  ← Agent system failure (AGT-6xxx)
    └── ConfigurationError          ← Missing env vars, bad config (CFG-8xxx)

Usage
-----
    from src.core.exceptions import IngestionError
    from src.core.error_codes import ErrorCode

    # Raise with code + message — both are logged and searchable
    raise IngestionError(
        error_code=ErrorCode.ING_FILE_NOT_FOUND,
        message=f"Raw file not found: {path}",
    )

    # Catch specifically — not bare Exception
    try:
        run_silver_pipeline()
    except SilverPipelineError as exc:
        logger.error("[%s] %s", exc.error_code, exc.message)
"""


class ClaimDenialSystemError(Exception):
    """
    Base exception for all errors raised by the Claim Denial System.

    Every custom exception in this system MUST inherit from this class.
    This allows callers to catch all system errors with a single handler
    if needed, while still being able to catch specific subclasses.

    Attributes:
        error_code: The error code string from ErrorCode (e.g. "ING-1001").
                    Used to correlate exception with the log line.
        message:    Human-readable description of what went wrong.
    """

    def __init__(self, error_code: str, message: str) -> None:
        """
        Args:
            error_code: Structured code from ErrorCode class (e.g. "ING-1001").
            message:    Plain-English description — do NOT hardcode this in
                        error handling; always pass the dynamic context.
        """
        # Store structured fields for programmatic access
        self.error_code: str = error_code
        self.message:    str = message

        # The string passed to the base Exception class appears in tracebacks.
        # Format: "[ING-1001] Raw file not found: /path/to/file"
        super().__init__(f"[{error_code}] {message}")


# --------------------------------------------------------------------------- #
#  Layer-specific exception subclasses                                          #
# --------------------------------------------------------------------------- #

class IngestionError(ClaimDenialSystemError):
    """
    Raised for failures in the Bronze ingestion layer.

    Use for:
    - Raw source file not found (ING-1001)
    - CSV schema mismatch — missing expected columns (ING-1002)
    - Bronze write failure — disk full or permissions (ING-1003)
    - Bronze table missing for downstream pipeline (ING-1004)
    """


class SilverPipelineError(ClaimDenialSystemError):
    """
    Raised for failures in the Silver cleaning or validation layer.

    Use for:
    - Null rate too high in critical columns (SLV-2001)
    - Critical join produced too many unexpected nulls (SLV-2004)
    - Input DataFrame is empty or has wrong schema (SLV-2005)
    """


class GoldPipelineError(ClaimDenialSystemError):
    """
    Raised for failures in the Gold feature engineering layer.

    Use for:
    - Validated Silver file not found (GLD-3001)
    - Feature engineering produced all-null column (GLD-3002)
    - Gold write failure (GLD-3004)
    """


class ModelNotFoundError(ClaimDenialSystemError):
    """
    Raised when a required ML model artifact file is missing.

    Use for:
    - denial_model.pkl not found (ML-4001)
    - threshold.json not found (ML-4001)
    - feature_medians.json not found (ML-4001)

    Typically means: run `python -m src.ml.trainer` first.
    """


class PredictionError(ClaimDenialSystemError):
    """
    Raised when ML model inference fails.

    Use for:
    - Feature schema mismatch — input columns don't match training (ML-4002)
    - Model returned invalid probability (ML-4003)
    - SHAP explainer failed (ML-4004)
    """


class RAGError(ClaimDenialSystemError):
    """
    Raised for failures in the RAG (Retrieval-Augmented Generation) system.

    Use for:
    - Policy document directory empty (RAG-5001)
    - FAISS index not found (RAG-5002)
    - LLM API call failed (RAG-5004)
    - Embedding model failure (RAG-5005)
    """


class AgentError(ClaimDenialSystemError):
    """
    Raised for failures in the claim remediation agent.

    Use for:
    - Invalid claim input to agent (AGT-6001)
    - Agent failed to produce remediation plan (AGT-6002)
    - Agent preprocessing produced no features (AGT-6003)
    """


class ConfigurationError(ClaimDenialSystemError):
    """
    Raised when required configuration or environment setup is missing.

    Use for:
    - Required environment variable not set (CFG-8001)
    - Invalid LOG_LEVEL value (CFG-8002)
    """
