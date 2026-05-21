"""
Week 1 — Bronze Layer: PySpark / Databricks Loader
----------------------------------------------------
Loads raw CSV files into Delta Lake Bronze tables.

NOTE: This file is designed to run on Databricks or a local Spark
      environment with Delta Lake configured.
      For local Mac development WITHOUT Spark, use local_loader.py instead.

Usage (Databricks):
  Run as a notebook cell or as part of a Databricks job.
Local usage (requires Java + PySpark installed):
  python -m src.ingestion.bronze_loader
"""

import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit
from delta import configure_spark_with_delta_pip

# --------------------------------------------------------------------------- #
#  Paths — resolved relative to project root regardless of CWD                 #
# --------------------------------------------------------------------------- #
PROJECT_ROOT  = Path(__file__).resolve().parents[2]   # claim-denial-system/
LOG_DIR       = PROJECT_ROOT / "logs"
DELTA_DIR     = PROJECT_ROOT / "data" / "delta"       # local Delta storage
RAW_DIR       = PROJECT_ROOT / "data" / "raw"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "ingestion_spark.log"),
    ],
)
logger = logging.getLogger(__name__)

# Dataset config: {raw filename → Delta table sub-path}
DATASETS: dict[str, str] = {
    "claims_1000.csv":    "bronze_claims",
    "providers_1000.csv": "bronze_providers",
    "diagnosis.csv":      "bronze_diagnosis",
    "cost.csv":           "bronze_cost",
}


def get_spark() -> SparkSession:
    """
    Create or retrieve the active SparkSession with Delta Lake support.

    Configures a local Delta warehouse directory so this works on Mac
    without a Hive metastore.
    """
    warehouse_dir = str(DELTA_DIR / "warehouse")
    builder = (
        SparkSession.builder
        .appName("ClaimDenial_BronzeIngestion_v1")
        # Enable Delta Lake SQL extensions
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Use a local warehouse dir — avoids Hive metastore dependency
        .config("spark.sql.warehouse.dir", warehouse_dir)
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def load_csv_to_bronze(spark: SparkSession, file_path: str, table_name: str) -> int:
    """
    Read a CSV file and write it as a Delta table in the Bronze layer.

    Adds two metadata columns:
      - ingestion_timestamp: Spark current timestamp.
      - source_file: original filename for lineage tracking.

    Uses save() to an explicit path instead of saveAsTable() to avoid
    requiring a Hive metastore.

    Args:
        spark:      Active SparkSession.
        file_path:  Absolute path to the source CSV.
        table_name: Name used for the output Delta sub-directory.

    Returns:
        Row count of the written table.

    Raises:
        FileNotFoundError: If the source CSV does not exist.
    """
    logger.info(f"Ingesting: {file_path} → {table_name}")

    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Source file not found: {file_path}")

    try:
        # Read CSV — inferSchema detects column types automatically
        df = (
            spark.read.csv(file_path, header=True, inferSchema=True)
            .withColumn("ingestion_timestamp", current_timestamp())
            .withColumn("source_file", lit(Path(file_path).name))
        )

        # Write to explicit Delta path — no Hive metastore required
        delta_path = str(DELTA_DIR / "bronze" / table_name)
        df.write.format("delta").mode("overwrite").save(delta_path)

        # Read back to count
        final_count = spark.read.format("delta").load(delta_path).count()
        logger.info(f"SUCCESS: {table_name} → {final_count:,} rows at {delta_path}")
        return final_count

    except Exception as e:
        logger.error(f"FAILED to load {table_name}: {e}")
        raise


def run_bronze_pipeline() -> dict[str, int]:
    """
    Run the complete Bronze ingestion pipeline for all configured datasets.

    Returns:
        Dict of {table_name: row_count}. Tables with missing source files
        are logged as warnings and recorded with count 0.
    """
    spark = get_spark()
    results: dict[str, int] = {}

    try:
        for filename, table_name in DATASETS.items():
            file_path = str(RAW_DIR / filename)
            try:
                count = load_csv_to_bronze(spark, file_path, table_name)
                results[table_name] = count
            except FileNotFoundError:
                logger.warning(f"Skipping {filename} — file not found in {RAW_DIR}")
                results[table_name] = 0

        logger.info(f"Pipeline complete. Results: {results}")
        return results

    finally:
        # Always stop Spark cleanly when running as a standalone script
        spark.stop()
        logger.info("SparkSession stopped.")


if __name__ == "__main__":
    results = run_bronze_pipeline()
    print("\nBronze Layer (Spark) Ingestion Complete:")
    for table, count in results.items():
        status = "✓ OK" if count > 0 else "⚠ 0 rows"
        print(f"  {table:<30} {count:>6,} rows  [{status}]")
