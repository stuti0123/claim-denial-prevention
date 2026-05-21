"""
Week 2 — Data Profiling: Multi-Dataset Quality Orchestrator
-------------------------------------------------------------
Loads all four Bronze tables and produces a combined data-quality
summary report, highlighting critical issues (missing key columns,
high null rates) across the entire dataset.

Output: docs/data_quality_summary.txt

Usage:
  python -m src.profiling.data_profiler
"""

import logging
from pathlib import Path

import pandas as pd

from src.profiling.profiler import DOCS_DIR, LOG_DIR, profile_dataframe

# --------------------------------------------------------------------------- #
#  Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR   = PROJECT_ROOT / "data" / "bronze"

LOG_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "profiling.log"),
    ],
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Expected critical columns per table (used for quality checks)               #
# --------------------------------------------------------------------------- #
QUALITY_CHECKS: dict[str, list[str]] = {
    "bronze_claims":    [
        "claim_id", "patient_id", "provider_id",
        "diagnosis_code", "procedure_code", "billed_amount",
    ],
    "bronze_providers": ["provider_id", "doctor_name", "specialty"],
    "bronze_diagnosis": ["diagnosis_code", "category", "severity"],
    "bronze_cost":      ["procedure_code", "average_cost", "expected_cost"],
}

# Null rate above this threshold triggers a warning in the summary
NULL_WARN_THRESHOLD_PCT = 10.0


def load_bronze_tables() -> dict[str, pd.DataFrame]:
    """
    Load all four Bronze CSV tables into DataFrames.

    Returns:
        Dict of {table_name: DataFrame}. Tables whose CSV is missing are
        logged as warnings and excluded from the result.
    """
    tables: dict[str, pd.DataFrame] = {}
    for table_name in QUALITY_CHECKS:
        path = BRONZE_DIR / f"{table_name}.csv"
        if path.exists():
            tables[table_name] = pd.read_csv(path)
            logger.info(f"Loaded {table_name}: {len(tables[table_name]):,} rows")
        else:
            logger.warning(f"Bronze table not found (run ingestion first): {path}")
    return tables


def check_quality(
    df: pd.DataFrame,
    table_name: str,
    expected_cols: list[str],
) -> list[str]:
    """
    Check a table for missing columns and high null rates.

    Args:
        df:            DataFrame to inspect.
        table_name:    Name used in issue messages.
        expected_cols: Columns that must be present and populated.

    Returns:
        List of human-readable issue strings found.
    """
    issues: list[str] = []
    for col in expected_cols:
        if col not in df.columns:
            issues.append(f"[{table_name}] COLUMN MISSING: '{col}'")
        else:
            null_pct = df[col].isnull().mean() * 100
            if null_pct > NULL_WARN_THRESHOLD_PCT:
                issues.append(
                    f"[{table_name}] HIGH NULLS in '{col}': {null_pct:.1f}%"
                )
    return issues


def run_data_profiling() -> None:
    """
    Orchestrate multi-table quality checks and save the combined summary.

    Steps:
      1. Load all Bronze tables.
      2. Run column-level quality checks on each.
      3. Print a combined summary with all issues flagged.
      4. Save the summary to docs/data_quality_summary.txt.
    """
    tables = load_bronze_tables()

    lines: list[str] = [
        "=" * 60,
        "DATA QUALITY SUMMARY — ALL BRONZE TABLES",
        "=" * 60,
        "",
    ]
    all_issues: list[str] = []

    for table_name, df in tables.items():
        expected_cols = QUALITY_CHECKS.get(table_name, [])
        issues = check_quality(df, table_name, expected_cols)
        all_issues.extend(issues)

        # Per-table summary row
        null_total = df.isnull().sum().sum()
        lines += [
            f"Table   : {table_name}",
            f"  Rows          : {len(df):,}",
            f"  Columns       : {len(df.columns)}",
            f"  Total nulls   : {null_total:,}",
            f"  Issues found  : {len(issues)}",
            "",
        ]

    # Aggregate issues section
    lines.append("--- Issues Requiring Attention ---")
    if all_issues:
        for issue in all_issues:
            lines.append(f"  ⚠  {issue}")
    else:
        lines.append("  ✓  No critical issues found.")

    summary_text = "\n".join(lines)
    print(summary_text)

    # Save to docs/
    out_path = DOCS_DIR / "data_quality_summary.txt"
    out_path.write_text(summary_text, encoding="utf-8")
    logger.info(f"Data quality summary saved → {out_path}")

    # Also run detailed per-table profiles and append them
    detailed = "\n\n--- DETAILED PER-TABLE PROFILES ---\n"
    for table_name, df in tables.items():
        detailed += profile_dataframe(df, f"{table_name}.csv")

    full_path = DOCS_DIR / "full_profiling_report.txt"
    full_path.write_text(summary_text + detailed, encoding="utf-8")
    logger.info(f"Full combined report saved → {full_path}")


if __name__ == "__main__":
    run_data_profiling()
