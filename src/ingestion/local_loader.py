"""
src/ingestion/local_loader.py
------------------------------
Week 1 — Bronze Layer: Local Pandas Loader

Loads raw CSV files from data/raw/ into the Bronze layer (data/bronze/).

Bronze layer rule:
  Only ADD metadata (ingestion_timestamp, source_file).
  Never transform, filter, or clean the data — that is Silver's job.

Logging:
  All output goes to logs/app.log via get_logger().
  Every error log line carries an error code (e.g. ING-1001) that can be
  searched with: grep "ING-1001" logs/app.log | wc -l

Exception handling:
  try/except blocks are placed at I/O boundaries (read, write) and at the
  top-level pipeline orchestrator. Inner pure functions raise typed
  IngestionError; the orchestrator catches and logs them.

Scale note:
  This loader uses pandas.read_csv() with default chunking. For files with
  1M+ records, replace pd.read_csv() with pd.read_csv(..., chunksize=N)
  and process chunks in a loop to avoid loading the full file into RAM.

Usage:
  python -m src.ingestion.local_loader
  OR import and call run_ingestion() from other modules.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Core infrastructure — centralised logging, error codes, custom exceptions
from src.core.logger import get_logger
from src.core.error_codes import ErrorCode
from src.core.exceptions import IngestionError

# --------------------------------------------------------------------------- #
#  Paths — resolved relative to this file so the module works from any cwd     #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # claim-denial-system/
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
BRONZE_DIR   = PROJECT_ROOT / "data" / "bronze"
LOG_DIR      = PROJECT_ROOT / "logs"

# --------------------------------------------------------------------------- #
#  Logging                                                                      #
# --------------------------------------------------------------------------- #
# Do NOT call logging.basicConfig() here — get_logger() handles it once.
logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Dataset config: {raw filename → bronze table name}                          #
# --------------------------------------------------------------------------- #
DATASETS: dict[str, str] = {
    "claims_1000.csv":    "bronze_claims",
    "providers_1000.csv": "bronze_providers",
    "diagnosis.csv":      "bronze_diagnosis",
    "cost.csv":           "bronze_cost",
}


# --------------------------------------------------------------------------- #
#  Functions                                                                    #
# --------------------------------------------------------------------------- #

def load_csv(file_name: str) -> Optional[pd.DataFrame]:
    """
    Load a single raw CSV file and stamp it with Bronze metadata.

    Only two columns are added (ingestion_timestamp, source_file) — no
    transformations. This is the Bronze layer contract.

    For 1M+ row files: consider replacing pd.read_csv() with chunked reading.

    Args:
        file_name: CSV filename inside data/raw/ (e.g. 'claims_1000.csv').

    Returns:
        DataFrame with ingestion_timestamp and source_file columns added.
        Returns None if the file does not exist (logged as warning, not error —
        a missing optional table should not crash the pipeline).

    Raises:
        IngestionError: If the file exists but cannot be read (corrupt, locked).
    """
    file_path = RAW_DIR / file_name

    # A missing file is a WARNING, not an error — skip it and continue
    if not file_path.exists():
        logger.warning(
            "[%s] File not found, skipping. file=%s",
            ErrorCode.ING_FILE_NOT_FOUND, file_name,
        )
        return None

    try:
        df = pd.read_csv(file_path)

    except Exception as exc:
        logger.error(
            "[%s] Failed to read CSV. file=%s error=%s",
            ErrorCode.ING_FILE_NOT_FOUND, file_name, str(exc),
        )
        raise IngestionError(
            error_code=ErrorCode.ING_FILE_NOT_FOUND,
            message=f"Failed to read CSV file '{file_name}': {exc}",
        ) from exc

    # Add Bronze metadata — the only two additions allowed at this layer
    df["ingestion_timestamp"] = datetime.now().isoformat()
    df["source_file"]         = file_name

    # Pass values as logger arguments — avoids f-string formatting cost
    # when the log level is above INFO (important for production performance)
    logger.info(
        "Loaded raw file. file=%s rows=%d columns=%d",
        file_name, len(df), len(df.columns),
    )
    return df


def save_bronze(df: pd.DataFrame, table_name: str) -> Path:
    """
    Persist a DataFrame as a Bronze CSV file.

    Args:
        df:         DataFrame to save (already stamped with metadata).
        table_name: Output base name (e.g. 'bronze_claims' → bronze_claims.csv).

    Returns:
        Path of the saved CSV file.

    Raises:
        IngestionError: If the file cannot be written (disk full, permissions).
    """
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BRONZE_DIR / f"{table_name}.csv"

    try:
        # I/O boundary: wrap disk write in try/except
        df.to_csv(out_path, index=False)

    except Exception as exc:
        logger.error(
            "[%s] Failed to write Bronze table. table=%s path=%s error=%s",
            ErrorCode.ING_WRITE_FAILED, table_name, out_path, str(exc),
        )
        raise IngestionError(
            error_code=ErrorCode.ING_WRITE_FAILED,
            message=f"Failed to write Bronze table '{table_name}' to {out_path}: {exc}",
        ) from exc

    logger.info(
        "Bronze table saved. table=%s path=%s rows=%d",
        table_name, out_path, len(df),
    )
    return out_path


def run_ingestion() -> dict[str, pd.DataFrame]:
    """
    Run the full Bronze ingestion pipeline for all configured datasets.

    Loads each raw CSV, adds metadata, and writes it to data/bronze/.
    Tables whose source file is missing are skipped with a WARNING (not an
    error) — this allows partial ingestion when some files are not available.

    Returns:
        Dict of {table_name: DataFrame} for every successfully loaded table.

    Raises:
        IngestionError: If a file exists but cannot be read or written.
                        Missing files are silently skipped (logged as warnings).
    """
    logger.info("=== Bronze Ingestion Started ===")
    results: dict[str, pd.DataFrame] = {}

    for file_name, table_name in DATASETS.items():
        # Each table is loaded independently — one failure does not stop others.
        # Missing files → None → skipped. Read/write errors → IngestionError raised.
        try:
            df = load_csv(file_name)
            if df is not None:
                save_bronze(df, table_name)
                results[table_name] = df

        except IngestionError:
            # Already logged inside load_csv / save_bronze — just re-raise
            # so the caller knows something went wrong.
            raise

    logger.info(
        "=== Bronze Ingestion Complete === loaded=%d/%d tables",
        len(results), len(DATASETS),
    )
    return results


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # APP_ENV guard: suppress console summary output in production.
    # Set APP_ENV=production in the deployment environment (.env or CI/CD secret)
    # to keep logs clean. In development (default), the summary is printed.
    is_production = os.getenv("APP_ENV", "development") == "production"

    try:
        data = run_ingestion()
    except IngestionError as exc:
        logger.error("Ingestion failed. code=%s error=%s", exc.error_code, exc.message)
        raise SystemExit(1) from exc

    if not is_production:
        # Development-only summary table — not shown in production
        print("\nBronze Layer Ingestion Summary:")
        print(f"  {'Table':<30} {'Rows':>6}  Status")
        print(f"  {'-'*30} {'-'*6}  ------")
        for name, df in data.items():
            print(f"  {name:<30} {len(df):>6,}  ✓ OK")
        missing = set(DATASETS.values()) - set(data.keys())
        for name in missing:
            print(f"  {name:<30} {'—':>6}  ⚠ FILE NOT FOUND")