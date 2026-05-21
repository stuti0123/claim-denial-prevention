"""
Unit Tests — Week 1: Bronze Layer (local_loader.py)
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# We test the functions directly rather than running the pipeline
from src.ingestion.local_loader import load_csv, save_bronze, DATASETS


# --------------------------------------------------------------------------- #
#  Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """Create a minimal temporary CSV to use as a fake raw file."""
    csv_content = (
        "claim_id,patient_id,billed_amount\n"
        "C001,P001,5000\n"
        "C002,P002,\n"          # missing billed_amount
        "C001,P001,5000\n"      # duplicate claim_id
    )
    csv_file = tmp_path / "test_claims.csv"
    csv_file.write_text(csv_content)
    return csv_file


# --------------------------------------------------------------------------- #
#  load_csv tests                                                               #
# --------------------------------------------------------------------------- #

def test_load_csv_adds_metadata(sample_csv: Path, monkeypatch) -> None:
    """Bronze metadata columns must be added to every loaded file."""
    # Point RAW_DIR to our temp dir
    monkeypatch.setattr("src.ingestion.local_loader.RAW_DIR", sample_csv.parent)

    df = load_csv(sample_csv.name)

    assert df is not None, "load_csv should return a DataFrame"
    assert "ingestion_timestamp" in df.columns, "ingestion_timestamp column missing"
    assert "source_file" in df.columns, "source_file column missing"


def test_load_csv_source_file_value(sample_csv: Path, monkeypatch) -> None:
    """source_file column should contain the original filename."""
    monkeypatch.setattr("src.ingestion.local_loader.RAW_DIR", sample_csv.parent)

    df = load_csv(sample_csv.name)
    assert (df["source_file"] == sample_csv.name).all()


def test_load_csv_returns_none_for_missing_file(tmp_path: Path, monkeypatch) -> None:
    """load_csv must return None (not raise) when the file does not exist."""
    monkeypatch.setattr("src.ingestion.local_loader.RAW_DIR", tmp_path)

    result = load_csv("nonexistent_file.csv")
    assert result is None


def test_load_csv_row_count(sample_csv: Path, monkeypatch) -> None:
    """All rows (including duplicates) must be loaded — no filtering in Bronze."""
    monkeypatch.setattr("src.ingestion.local_loader.RAW_DIR", sample_csv.parent)

    df = load_csv(sample_csv.name)
    assert len(df) == 3  # 3 data rows (duplicate kept — Bronze doesn't clean)


# --------------------------------------------------------------------------- #
#  save_bronze tests                                                            #
# --------------------------------------------------------------------------- #

def test_save_bronze_creates_file(tmp_path: Path, monkeypatch) -> None:
    """save_bronze must write a CSV to the Bronze directory."""
    monkeypatch.setattr("src.ingestion.local_loader.BRONZE_DIR", tmp_path)

    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out_path = save_bronze(df, "test_table")

    assert out_path.exists(), "Bronze CSV was not created"


def test_save_bronze_content(tmp_path: Path, monkeypatch) -> None:
    """The saved Bronze CSV must round-trip correctly."""
    monkeypatch.setattr("src.ingestion.local_loader.BRONZE_DIR", tmp_path)

    df = pd.DataFrame({"claim_id": ["C001", "C002"], "amount": [100, 200]})
    out_path = save_bronze(df, "roundtrip_table")

    reloaded = pd.read_csv(out_path)
    assert list(reloaded["claim_id"]) == ["C001", "C002"]
    assert list(reloaded["amount"])   == [100, 200]
