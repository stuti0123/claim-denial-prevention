"""
src/silver/validator.py
-----------------------
Week 3 — Silver Layer: Rule-Based Claim Validator

Applies business validation rules to the Silver claims table. Each rule adds
a boolean flag column to the DataFrame. A combined 'validation_flags' column
lists all fired rule names as a pipe-separated string.
An empty string means the claim passed all checks.

Rules:
  Rule 1 — WARN_MISSING_DIAGNOSIS : diagnosis_code is null/empty
  Rule 2 — WARN_MISSING_PROCEDURE : procedure_code is null/empty
  Rule 3 — WARN_MISSING_AMOUNT    : billed_amount is null
  Rule 4 — WARN_HIGH_BILLING      : billed > HIGH_BILLING_MULTIPLIER × expected
  Rule 5 — WARN_INVALID_DIAGNOSIS : code present but not in reference set
  Rule 6 — ERR_INCOMPLETE_CLAIM   : BOTH diagnosis AND procedure are missing
                                    (most severe — likely to be denied)

Logging:
  All output goes to logs/app.log via get_logger().
  Error codes (SLV-2xxx) make log lines searchable by category.

Exception handling:
  try/except is placed at the top-level run_validation() entry point.
  Inner rule functions are pure DataFrame operations — they don't do I/O and
  therefore do not need individual try/except blocks.

Output: data/silver/silver_claims_validated.csv

Usage:
  python -m src.silver.validator
  OR: from src.silver.validator import run_validation
"""

import os
from pathlib import Path

import pandas as pd

from src.silver.cleaner import run_silver_pipeline, SILVER_DIR

# Core infrastructure — centralised logging, error codes, custom exceptions
from src.core.logger import get_logger
from src.core.error_codes import ErrorCode
from src.core.exceptions import IngestionError, SilverPipelineError

# --------------------------------------------------------------------------- #
#  Logging                                                                      #
# --------------------------------------------------------------------------- #
# NOTE: get_logger() configures the root logger ONCE on first call. All modules
# across the system share the same root logger — logs go to logs/app.log.
# Do NOT call logging.basicConfig() anywhere in this codebase.
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Constants                                                                    #
# --------------------------------------------------------------------------- #

# Valid ICD-style diagnosis codes from the reference table (diagnosis.csv)
# frozenset: O(1) membership test — important for 1M+ rows
VALID_DIAGNOSIS_CODES: frozenset[str] = frozenset(
    {"D10", "D20", "D30", "D40", "D50", "D60"}
)

# A billed amount more than this multiple of expected_cost triggers a flag
HIGH_BILLING_MULTIPLIER: float = 3.0

# Ordered list of (flag_column_name, rule_label) used to build the summary.
# Order matters: ERR rules are listed last for display clarity.
RULE_LABELS: list[tuple[str, str]] = [
    ("flag_missing_diagnosis", "WARN_MISSING_DIAGNOSIS"),
    ("flag_missing_procedure", "WARN_MISSING_PROCEDURE"),
    ("flag_missing_amount",    "WARN_MISSING_AMOUNT"),
    ("flag_high_billing",      "WARN_HIGH_BILLING"),
    ("flag_invalid_diagnosis", "WARN_INVALID_DIAGNOSIS"),
    ("flag_incomplete_claim",  "ERR_INCOMPLETE_CLAIM"),
]


# --------------------------------------------------------------------------- #
#  Validation engine                                                            #
# --------------------------------------------------------------------------- #

def apply_validation_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all business rules and produce per-claim flag columns.

    Each rule adds one boolean column. A combined 'validation_flags' string
    column is built at the end by concatenating all fired rule names.

    WHY NO try/except HERE?
    This function is a pure data transformation — it reads and writes columns
    on an in-memory DataFrame. Pandas operations here do not do I/O and
    cannot fail due to external resources. The top-level run_validation()
    function wraps the entire call in try/except.

    Args:
        df: Enriched Silver claims DataFrame (output of cleaner.py).

    Returns:
        DataFrame with one boolean flag column per rule and a combined
        'validation_flags' string column.
    """
    df = df.copy()   # defensive copy — never mutate the caller's DataFrame

    # Rule 1: diagnosis_code is null
    df["flag_missing_diagnosis"] = df["diagnosis_code"].isna()

    # Rule 2: procedure_code is null
    df["flag_missing_procedure"] = df["procedure_code"].isna()

    # Rule 3: billed_amount is null
    df["flag_missing_amount"] = df["billed_amount"].isna()

    # Rule 4: billed_amount > 3× expected_cost (only when both values exist)
    # Guard: only compare when both columns have non-null values
    df["flag_high_billing"] = (
        df["billed_amount"].notna()
        & df["expected_cost"].notna()
        & (df["billed_amount"] > HIGH_BILLING_MULTIPLIER * df["expected_cost"])
    )

    # Rule 5: diagnosis_code is present but not in the reference list
    df["flag_invalid_diagnosis"] = (
        df["diagnosis_code"].notna()
        & ~df["diagnosis_code"].isin(VALID_DIAGNOSIS_CODES)
    )

    # Rule 6 (most severe): BOTH diagnosis AND procedure are missing
    # This is the strongest predictor of denial — missing both fields means
    # the claim cannot be adjudicated at all.
    df["flag_incomplete_claim"] = (
        df["flag_missing_diagnosis"] & df["flag_missing_procedure"]
    )

    # Build pipe-separated summary string of all fired rules per claim.
    # Each claim gets either "" (clean) or e.g. "WARN_MISSING_DIAGNOSIS|ERR_INCOMPLETE_CLAIM".
    df["validation_flags"] = df.apply(_build_flag_string, axis=1)

    return df


def _build_flag_string(row: pd.Series) -> str:
    """
    Concatenate the labels of all fired rules for one claim row.

    WHY underscore prefix?
    Leading underscore means this is an implementation detail of this module.
    It is called via DataFrame.apply() internally — not a public API function.

    Args:
        row: A single row from the validated DataFrame.

    Returns:
        Pipe-separated string of rule labels, or '' if no rules fired.
        Example: "WARN_MISSING_DIAGNOSIS|WARN_HIGH_BILLING"
    """
    fired = [
        label
        for flag_col, label in RULE_LABELS
        if row.get(flag_col, False)
    ]
    return "|".join(fired)


def summarise_flags(df: pd.DataFrame) -> None:
    """
    Log a concise summary of how many claims triggered each validation rule.

    Args:
        df: Validated DataFrame (must have the flag columns and validation_flags).
    """
    total = len(df)
    logger.info("--- Validation Summary (%d claims) ---", total)
    logger.info("  %-30s %6s  %12s", "Rule", "Count", "% of Claims")
    logger.info("  %s %s  %s", "-" * 30, "-" * 6, "-" * 12)

    for flag_col, label in RULE_LABELS:
        if flag_col in df.columns:
            count = int(df[flag_col].sum())   # explicit int — bool sum returns numpy int
            pct   = count / total * 100
            logger.info("  %-30s %6d  %11.1f%%", label, count, pct)

    clean = (df["validation_flags"] == "").sum()
    logger.info("  %-30s %6d  %11.1f%%", "CLEAN CLAIMS (no flags)", clean, clean / total * 100)
    logger.info("--- End of Validation Summary ---")


# --------------------------------------------------------------------------- #
#  Pipeline entry-point                                                         #
# --------------------------------------------------------------------------- #

def run_validation(silver_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Run validation rules on the Silver claims table.

    If silver_df is not provided, runs the Silver pipeline first to generate
    the input data. This enables calling run_validation() standalone.

    Args:
        silver_df: Pre-loaded Silver DataFrame (optional). If None,
                   run_silver_pipeline() is called automatically.

    Returns:
        Validated DataFrame with flag columns and 'validation_flags' column.

    Raises:
        IngestionError: If the Silver pipeline fails due to missing Bronze tables.
        SilverPipelineError: If validation or saving fails.
    """
    try:
        if silver_df is None:
            logger.debug(
                "[%s] No Silver DataFrame provided — running Silver pipeline first.",
                ErrorCode.SLV_HIGH_NULL_RATE,
            )
            silver_df = run_silver_pipeline()

        logger.info(
            "Applying validation rules. claims=%d",
            len(silver_df),
        )
        validated_df = apply_validation_rules(silver_df)

        # Count and log how many claims have at least one flag
        flagged = (validated_df["validation_flags"] != "").sum()
        logger.info(
            "Validation complete. flagged=%d total=%d",
            flagged, len(validated_df),
        )

        # Save validated Silver table — wrap disk write in try/except
        out_path = SILVER_DIR / "silver_claims_validated.csv"
        try:
            validated_df.to_csv(out_path, index=False)
        except Exception as exc:
            logger.error(
                "[%s] Failed to save validated Silver table. path=%s error=%s",
                ErrorCode.GLD_WRITE_FAILED, out_path, str(exc),
            )
            raise SilverPipelineError(
                error_code=ErrorCode.GLD_WRITE_FAILED,
                message=f"Failed to write validated Silver table to {out_path}: {exc}",
            ) from exc

        logger.info("Validated Silver table saved. path=%s", out_path)
        return validated_df

    except (IngestionError, SilverPipelineError):
        # Already logged — just propagate
        raise
    except Exception as exc:
        # Unexpected error — wrap for consistent handling
        logger.error(
            "[%s] Unexpected error in validation pipeline. error=%s",
            ErrorCode.SLV_HIGH_NULL_RATE, str(exc),
        )
        raise SilverPipelineError(
            error_code=ErrorCode.SLV_HIGH_NULL_RATE,
            message=f"Unexpected validation pipeline error: {exc}",
        ) from exc


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # APP_ENV guard: summarise_flags logs to app.log in all environments.
    # No print() calls here — all output goes through the logger.
    is_production = os.getenv("APP_ENV", "development") == "production"

    try:
        validated = run_validation()
        summarise_flags(validated)
    except (IngestionError, SilverPipelineError) as exc:
        logger.error("Validation failed. code=%s error=%s", exc.error_code, exc.message)
        raise SystemExit(1) from exc

    if not is_production:
        # Development only: show flagged count summary in console
        flagged = (validated["validation_flags"] != "").sum()
        print(f"\nValidation Complete: {flagged:,}/{len(validated):,} claims flagged.")
        print("Full summary written to logs/app.log")
