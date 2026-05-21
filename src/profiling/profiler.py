"""
Week 2 — Data Profiling: Single-Table Profiler
-----------------------------------------------
Profiles individual Bronze CSV files and produces a text report covering:
  - Shape (rows, columns)
  - Missing values per column (count + %)
  - Duplicate check on claim_id (when present)
  - Date range for any date columns
  - Column cardinality (unique value counts)
  - Numeric statistics (describe)

Saves the full report to docs/profiling_report.txt.

Usage:
  python -m src.profiling.profiler
"""

import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
#  Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR   = PROJECT_ROOT / "data" / "bronze"
DOCS_DIR     = PROJECT_ROOT / "docs"
LOG_DIR      = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "profiling.log"),
    ],
)
logger = logging.getLogger(__name__)


def profile_dataframe(df: pd.DataFrame, file_name: str) -> str:
    """
    Generate a structured text profile report for one DataFrame.

    Args:
        df:        DataFrame to analyse.
        file_name: Source file name shown in the report header.

    Returns:
        Multi-line string containing the full profile report.
    """
    buf = StringIO()

    def w(line: str = "") -> None:
        """Write a line to the buffer."""
        buf.write(line + "\n")

    w("=" * 60)
    w(f"PROFILING REPORT: {file_name}")
    w(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 60)

    # Basic shape
    w(f"\nRows    : {len(df):,}")
    w(f"Columns : {len(df.columns)}")
    w(f"Names   : {list(df.columns)}")

    # --- Missing values ---
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    w("\n--- Missing Values ---")
    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            w(f"  {col:<30} {missing[col]:>5} missing  ({missing_pct[col]}%)")
            has_missing = True
    if not has_missing:
        w("  ✓ No missing values")

    # --- Duplicate check (only for tables with claim_id) ---
    if "claim_id" in df.columns:
        dups = df.duplicated(subset=["claim_id"]).sum()
        w(f"\n--- Duplicate claim_ids ---")
        w(f"  {dups} duplicate(s) found")

    # --- Date range check ---
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for col in date_cols:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            w(f"\n--- Date Range ({col}) ---")
            
            # Check if parsed is a Series (has min/max as callables) or a scalar
            if isinstance(parsed, pd.Series):
                min_val = parsed.min()
                max_val = parsed.max()
                invalid_count = parsed.isna().sum()
            else:
                min_val = parsed
                max_val = parsed
                invalid_count = 1 if pd.isna(parsed) else 0

            w(f"  Min     : {min_val}")
            w(f"  Max     : {max_val}")
            w(f"  Invalid : {invalid_count} unparseable date(s)")
        except Exception:
            pass

    # --- Column cardinality ---
    w("\n--- Column Cardinality (unique values) ---")
    for col in df.columns:
        w(f"  {col:<30} {df[col].nunique():>5} unique")

    # --- Numeric statistics ---
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        w("\n--- Numeric Statistics ---")
        w(df[num_cols].describe().round(2).to_string())

    w("")  # trailing blank line
    return buf.getvalue()


def profile_file(file_name: str) -> Optional[str]:
    """
    Load one Bronze CSV file and return its profile report string.

    Args:
        file_name: CSV filename inside data/bronze/ (e.g. 'bronze_claims.csv').

    Returns:
        Profile report string, or None if the file is missing.
    """
    file_path = BRONZE_DIR / file_name

    if not file_path.exists():
        logger.warning(f"File not found — skipping: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Profiling {file_name} ({len(df):,} rows)")
        return profile_dataframe(df, file_name)
    except Exception as e:
        logger.error(f"Failed to profile {file_name}: {e}")
        return None


def run_profiling() -> None:
    """
    Profile all Bronze CSV files in data/bronze/ and save the report.

    Prints each report to the console and writes the combined report
    to docs/profiling_report.txt.
    """
    csv_files = sorted(BRONZE_DIR.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {BRONZE_DIR}. Run ingestion first.")
        return

    full_report = ""
    for csv_file in csv_files:
        report = profile_file(csv_file.name)
        if report:
            print(report)
            full_report += report

    # Persist report for future reference / CI checks
    report_path = DOCS_DIR / "profiling_report.txt"
    report_path.write_text(full_report, encoding="utf-8")
    logger.info(f"Full profiling report saved → {report_path}")


if __name__ == "__main__":
    run_profiling()