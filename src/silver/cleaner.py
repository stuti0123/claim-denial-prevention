"""
src/silver/cleaner.py
---------------------
Week 3 — Silver Layer: Data Cleaning & Enrichment

Reads all four Bronze tables, cleans and standardizes each one, then joins
them into a single enriched Silver claims table.

Silver layer principles:
  - Keep ALL rows (even incomplete ones) — dropping is Gold's job.
  - Standardize formats (dates, codes, strings).
  - Join reference data so downstream layers have one flat table.
  - Never add business logic — that is the validator's job.

Logging:
  All output goes to logs/app.log via get_logger().
  Error codes (SLV-2xxx) make log lines searchable by category.

Exception handling:
  try/except is placed at I/O boundaries (_load_bronze) and at the
  top-level pipeline entry point (run_silver_pipeline). Inner cleaning
  functions raise SilverPipelineError on unrecoverable schema violations.

Scale note:
  For 1M+ records, the merge operations in enrich_claims() remain efficient
  because pandas uses hash joins internally. Memory can be reduced further
  by casting string columns to Categorical dtype before the merge.

Output: data/silver/silver_claims.csv

Usage:
  python -m src.silver.cleaner
"""

import os
from pathlib import Path

import pandas as pd

# Core infrastructure — centralised logging, error codes, custom exceptions
from src.core.logger import get_logger
from src.core.error_codes import ErrorCode
from src.core.exceptions import IngestionError, SilverPipelineError

# --------------------------------------------------------------------------- #
#  Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR   = PROJECT_ROOT / "data" / "bronze"
SILVER_DIR   = PROJECT_ROOT / "data" / "silver"
LOG_DIR      = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
SILVER_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Logging                                                                      #
# --------------------------------------------------------------------------- #
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Private helpers (prefix _ = not part of public API)                         #
# --------------------------------------------------------------------------- #

def _load_bronze(table_name: str) -> pd.DataFrame:
    """
    Load a Bronze CSV table from disk.

    WHY underscore prefix?
    The leading underscore signals to other developers that this function is
    an internal implementation detail of this module, not a public API. It
    should not be imported or called from outside src/silver/cleaner.py.

    Args:
        table_name: Bronze table name without extension (e.g. 'bronze_claims').

    Returns:
        DataFrame loaded from data/bronze/<table_name>.csv.

    Raises:
        IngestionError: If the Bronze table does not exist.
        SilverPipelineError: If the file exists but cannot be read.
    """
    path = BRONZE_DIR / f"{table_name}.csv"

    if not path.exists():
        # Bronze table missing — this is an ingestion problem, use ING error code
        logger.error(
            "[%s] Bronze table not found. table=%s path=%s",
            ErrorCode.ING_BRONZE_MISSING, table_name, path,
        )
        raise IngestionError(
            error_code=ErrorCode.ING_BRONZE_MISSING,
            message=f"Bronze table not found: '{table_name}'. Run local_loader.py first.",
        )

    try:
        # I/O boundary: wrap file read in try/except
        df = pd.read_csv(path)

    except Exception as exc:
        logger.error(
            "[%s] Failed to read Bronze table. table=%s error=%s",
            ErrorCode.ING_BRONZE_MISSING, table_name, str(exc),
        )
        raise SilverPipelineError(
            error_code=ErrorCode.ING_BRONZE_MISSING,
            message=f"Failed to read Bronze table '{table_name}': {exc}",
        ) from exc

    logger.info(
        "Loaded bronze table. name=%s rows=%d",
        table_name, len(df),
    )
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase all column names and replace spaces with underscores.

    WHY: ensures consistent column access across all source files regardless
    of how the original CSV header was formatted.
    """
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
#  Per-table cleaning functions                                                 #
# --------------------------------------------------------------------------- #

def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw claims table.

    Steps:
      1. Normalize column names.
      2. Drop exact duplicate claim_ids (keep first — pipeline re-run safety).
      3. Parse the 'date' column to datetime; store as 'claim_date'.
      4. Uppercase medical code columns for consistent joins.
      5. Strip whitespace from all string columns.

    Args:
        df: Raw Bronze claims DataFrame.

    Returns:
        Cleaned claims DataFrame.
    """
    df = _normalize_columns(df)
    original_len = len(df)

    # Drop duplicate claim_ids — keep first occurrence
    df = df.drop_duplicates(subset=["claim_id"], keep="first")
    dropped = original_len - len(df)
    if dropped > 0:
        # SLV-2002: duplicate claim_ids found — pipeline re-run or data quality issue
        logger.warning(
            "[%s] Removed duplicate claim_ids. count=%d",
            ErrorCode.SLV_DUPLICATE_CLAIMS, dropped,
        )

    # Convert 'date' string → typed datetime column.
    # errors="coerce" converts unparseable dates to NaT instead of crashing.
    if "date" in df.columns:
        df["claim_date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.drop(columns=["date"])

        # Log if date parsing produced any NaT values (data quality signal)
        nat_count = df["claim_date"].isna().sum()
        if nat_count > 0:
            logger.warning(
                "[%s] Date parsing produced NaT for %d rows.",
                ErrorCode.SLV_DATE_PARSE_FAILED, nat_count,
            )

    # Uppercase medical codes for safe joins with reference tables
    for col in ["diagnosis_code", "procedure_code"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()

    # Strip leading/trailing whitespace from all text columns.
    # Use include="str" (pandas 3 style) — "object" is deprecated in pandas 3.
    str_cols = df.select_dtypes(include="str").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    logger.info(
        "Claims cleaned. original=%d final=%d rows",
        original_len, len(df),
    )
    return df


def clean_providers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the providers reference table.

    - Fills missing location with 'Unknown' (known data quality issue in source).
    - Strips whitespace from all string columns.

    Args:
        df: Raw Bronze providers DataFrame.

    Returns:
        Cleaned providers DataFrame.
    """
    df = _normalize_columns(df)
    df["location"] = df["location"].fillna("Unknown").str.strip()

    # Use include="str" for pandas 3 compatibility
    str_cols = df.select_dtypes(include="str").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    logger.info("Providers cleaned. rows=%d", len(df))
    return df


def clean_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the diagnosis reference table.

    - Uppercase diagnosis codes so joins work regardless of source casing.

    Args:
        df: Raw Bronze diagnosis DataFrame.

    Returns:
        Cleaned diagnosis DataFrame.
    """
    df = _normalize_columns(df)
    df["diagnosis_code"] = df["diagnosis_code"].str.strip().str.upper()
    df["severity"]       = df["severity"].str.strip()
    df["category"]       = df["category"].str.strip()
    logger.info("Diagnosis reference cleaned. rows=%d", len(df))
    return df


def clean_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the procedure cost reference table.

    - Uppercase procedure codes.
    - Coerce cost columns to float (handles any stray strings from source).

    Args:
        df: Raw Bronze cost DataFrame.

    Returns:
        Cleaned cost DataFrame.
    """
    df = _normalize_columns(df)
    df["procedure_code"] = df["procedure_code"].str.strip().str.upper()
    # errors="coerce" converts non-numeric strings to NaN instead of crashing
    df["average_cost"]   = pd.to_numeric(df["average_cost"],  errors="coerce")
    df["expected_cost"]  = pd.to_numeric(df["expected_cost"], errors="coerce")
    logger.info("Cost reference cleaned. rows=%d", len(df))
    return df


# --------------------------------------------------------------------------- #
#  Enrichment — join all four tables into one flat Silver table                 #
# --------------------------------------------------------------------------- #

def enrich_claims(
    claims:    pd.DataFrame,
    providers: pd.DataFrame,
    diagnosis: pd.DataFrame,
    cost:      pd.DataFrame,
) -> pd.DataFrame:
    """
    Join claims with the three reference tables to create one flat table.

    All joins are LEFT joins so every claim row is preserved even when
    a reference code is missing (those appear as NaN in reference columns).

    Scale note: For 1M+ rows, these pandas merges stay efficient because
    they use hash joins internally. No sorting is needed.

    Joins performed:
      claims + providers → adds doctor_name, specialty, location
      claims + diagnosis → adds category, severity
      claims + cost      → adds average_cost, expected_cost

    Args:
        claims:    Cleaned claims DataFrame.
        providers: Cleaned providers DataFrame.
        diagnosis: Cleaned diagnosis DataFrame.
        cost:      Cleaned cost DataFrame.

    Returns:
        Enriched DataFrame with all reference columns merged in.
    """
    # Bring in provider details
    df = claims.merge(
        providers[["provider_id", "doctor_name", "specialty", "location"]],
        on="provider_id",
        how="left",
    )

    # Bring in diagnosis category and severity
    df = df.merge(
        diagnosis[["diagnosis_code", "category", "severity"]],
        on="diagnosis_code",
        how="left",
    )

    # Bring in procedure cost benchmarks
    df = df.merge(
        cost[["procedure_code", "average_cost", "expected_cost"]],
        on="procedure_code",
        how="left",
    )

    logger.info(
        "Claims enriched. rows=%d columns=%d",
        len(df), len(df.columns),
    )
    return df


# --------------------------------------------------------------------------- #
#  Pipeline entry-point                                                         #
# --------------------------------------------------------------------------- #

def run_silver_pipeline() -> pd.DataFrame:
    """
    Execute the full Silver layer pipeline.

    Steps:
      1. Load Bronze tables.
      2. Clean each table.
      3. Join into one enriched Silver claims table.
      4. Save to data/silver/silver_claims.csv.

    Returns:
        Final Silver claims DataFrame.

    Raises:
        IngestionError: If a Bronze table is missing.
        SilverPipelineError: If cleaning or enrichment fails.
    """
    logger.info("=== Silver Pipeline Started ===")

    try:
        # Load — each call raises IngestionError if table is missing
        claims    = _load_bronze("bronze_claims")
        providers = _load_bronze("bronze_providers")
        diagnosis = _load_bronze("bronze_diagnosis")
        cost      = _load_bronze("bronze_cost")

        # Clean — pure transformations, should not raise unless data is corrupt
        claims    = clean_claims(claims)
        providers = clean_providers(providers)
        diagnosis = clean_diagnosis(diagnosis)
        cost      = clean_cost(cost)

        # Enrich — join all four tables
        silver_df = enrich_claims(claims, providers, diagnosis, cost)

        # Save — I/O boundary: wrap in try/except
        out_path = SILVER_DIR / "silver_claims.csv"
        try:
            silver_df.to_csv(out_path, index=False)
        except Exception as exc:
            logger.error(
                "[%s] Failed to save Silver table. path=%s error=%s",
                ErrorCode.GLD_WRITE_FAILED, out_path, str(exc),
            )
            raise SilverPipelineError(
                error_code=ErrorCode.GLD_WRITE_FAILED,
                message=f"Failed to write Silver table to {out_path}: {exc}",
            ) from exc

        logger.info("Silver table saved. path=%s", out_path)
        logger.info("=== Silver Pipeline Complete ===")
        return silver_df

    except (IngestionError, SilverPipelineError):
        # Already logged — just propagate so caller can handle
        raise
    except Exception as exc:
        # Unexpected error — wrap in SilverPipelineError for consistent handling
        logger.error(
            "[%s] Unexpected error in Silver pipeline. error=%s",
            ErrorCode.SLV_HIGH_NULL_RATE, str(exc),
        )
        raise SilverPipelineError(
            error_code=ErrorCode.SLV_HIGH_NULL_RATE,
            message=f"Unexpected Silver pipeline error: {exc}",
        ) from exc


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # APP_ENV guard: suppress summary output in production
    is_production = os.getenv("APP_ENV", "development") == "production"

    try:
        df = run_silver_pipeline()
    except (IngestionError, SilverPipelineError) as exc:
        logger.error("Silver pipeline failed. code=%s error=%s", exc.error_code, exc.message)
        raise SystemExit(1) from exc

    if not is_production:
        print(f"\nSilver Pipeline Complete:")
        print(f"  Rows    : {len(df):,}")
        print(f"  Columns : {len(df.columns)}")
        print(f"  Columns : {list(df.columns)}")
