"""
src/core/error_codes.py
-----------------------
Centralised catalogue of all error codes used across the Claim Denial System.

WHY ERROR CODES?
----------------
A code like "ING-1001" can be grepped across all log files in one command:

    grep "ING-1001" logs/app.log | wc -l

Without codes, you'd have to search free-text strings which break the moment
someone changes a log message.

CODE STRUCTURE
--------------
Each code has 3 parts:
    <LAYER>-<CATEGORY><SEQUENCE>

Layers:
    ING  — Ingestion (Bronze layer)
    SLV  — Silver layer (cleaning & validation)
    GLD  — Gold layer (feature engineering)
    ML   — Machine learning (training & prediction)
    RAG  — Retrieval-Augmented Generation
    AGT  — Agent system
    API  — FastAPI backend
    CFG  — Configuration & environment

Usage
-----
    from src.core.error_codes import ErrorCode

    logger.error("[%s] File not found: path=%s", ErrorCode.ING_FILE_NOT_FOUND, path)

    # Produces log line:
    # 2026-05-18 12:00:00 | ERROR | [ING-1001] | src.ingestion.local_loader | File not found: path=...
"""


class ErrorCode:
    """
    Namespace class holding all error code string constants.

    Organised by system layer. Each constant is a short string that can be
    embedded in any log message and searched across the unified log file.

    No instances of this class should be created — use the class attributes
    directly: ErrorCode.ING_FILE_NOT_FOUND
    """

    # ---------------------------------------------------------------------- #
    #  ING — Ingestion / Bronze layer (1xxx)                                   #
    # ---------------------------------------------------------------------- #

    # A raw source file expected in data/raw/ does not exist
    ING_FILE_NOT_FOUND      = "ING-1001"

    # A loaded CSV is missing one or more expected column names
    ING_SCHEMA_MISMATCH     = "ING-1002"

    # Could not write the Bronze CSV to disk
    ING_WRITE_FAILED        = "ING-1003"

    # A Bronze table that downstream layers depend on was not found
    ING_BRONZE_MISSING      = "ING-1004"

    # ---------------------------------------------------------------------- #
    #  SLV — Silver layer (2xxx)                                               #
    # ---------------------------------------------------------------------- #

    # Null rate in a critical column exceeds acceptable threshold
    SLV_HIGH_NULL_RATE      = "SLV-2001"

    # Duplicate claim_ids found and removed
    SLV_DUPLICATE_CLAIMS    = "SLV-2002"

    # Date parsing produced NaT for one or more rows
    SLV_DATE_PARSE_FAILED   = "SLV-2003"

    # A required Silver join produced unexpected nulls in merged columns
    SLV_JOIN_NULLS          = "SLV-2004"

    # Validation flagged a claim with the most severe rule (ERR_INCOMPLETE_CLAIM)
    SLV_INCOMPLETE_CLAIM    = "SLV-2005"

    # ---------------------------------------------------------------------- #
    #  GLD — Gold layer (3xxx)                                                 #
    # ---------------------------------------------------------------------- #

    # The validated Silver CSV was not found when Gold pipeline started
    GLD_SILVER_MISSING      = "GLD-3001"

    # A feature engineering step produced unexpected all-null output
    GLD_FEATURE_ALL_NULL    = "GLD-3002"

    # Denial label distribution is extreme (>95% or <5% denied)
    GLD_LABEL_IMBALANCE     = "GLD-3003"

    # Gold CSV could not be written
    GLD_WRITE_FAILED        = "GLD-3004"

    # ---------------------------------------------------------------------- #
    #  ML — Machine learning (4xxx)                                            #
    # ---------------------------------------------------------------------- #

    # Trained model file not found at expected path
    ML_MODEL_NOT_FOUND      = "ML-4001"

    # Model input features do not match training schema
    ML_FEATURE_MISMATCH     = "ML-4002"

    # Model returned a probability outside [0, 1]
    ML_INVALID_PROBABILITY  = "ML-4003"

    # SHAP explainer failed to compute attributions
    ML_SHAP_FAILED          = "ML-4004"

    # Training data has insufficient rows to train reliably
    ML_INSUFFICIENT_DATA    = "ML-4005"

    # ---------------------------------------------------------------------- #
    #  RAG — Retrieval-Augmented Generation (5xxx)                             #
    # ---------------------------------------------------------------------- #

    # Policy document directory is empty or missing
    RAG_NO_DOCUMENTS        = "RAG-5001"

    # FAISS index file not found — needs to be built first
    RAG_INDEX_NOT_FOUND     = "RAG-5002"

    # Vector search returned 0 results (index may be empty)
    RAG_EMPTY_RESULTS       = "RAG-5003"

    # LLM API call failed (timeout, rate limit, etc.)
    RAG_LLM_CALL_FAILED     = "RAG-5004"

    # Embedding model failed to load or process text
    RAG_EMBED_FAILED        = "RAG-5005"

    # ---------------------------------------------------------------------- #
    #  AGT — Agent system (6xxx)                                               #
    # ---------------------------------------------------------------------- #

    # Agent received a claim dict missing required fields
    AGT_INVALID_INPUT       = "AGT-6001"

    # Agent could not produce a remediation plan after all retries
    AGT_PLAN_FAILED         = "AGT-6002"

    # Agent preprocessing step produced no features
    AGT_PREPROCESS_FAILED   = "AGT-6003"

    # ---------------------------------------------------------------------- #
    #  API — FastAPI backend (7xxx)                                            #
    # ---------------------------------------------------------------------- #

    # A request payload failed Pydantic schema validation
    API_VALIDATION_ERROR    = "API-7001"

    # A requested claim_id does not exist in the data
    API_CLAIM_NOT_FOUND     = "API-7002"

    # An unhandled internal exception reached the route handler
    API_INTERNAL_ERROR      = "API-7003"

    # ---------------------------------------------------------------------- #
    #  CFG — Configuration / environment (8xxx)                                #
    # ---------------------------------------------------------------------- #

    # A required environment variable is missing
    CFG_ENV_VAR_MISSING     = "CFG-8001"

    # LOG_LEVEL environment variable has an invalid value
    CFG_INVALID_LOG_LEVEL   = "CFG-8002"
