"""
Unit Tests — Week 3: Silver Layer (cleaner.py + validator.py)
"""

import pandas as pd
import pytest

from src.silver.cleaner   import clean_claims, clean_cost, enrich_claims
from src.silver.validator import apply_validation_rules
from src.core.exceptions  import IngestionError, SilverPipelineError


# --------------------------------------------------------------------------- #
#  Fixtures — minimal DataFrames matching the real schema                       #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def raw_claims() -> pd.DataFrame:
    """Minimal raw claims with duplicates and missing values."""
    return pd.DataFrame({
        "claim_id":        ["C001", "C002", "C003", "C001"],  # C001 duplicated
        "patient_id":      ["P001", "P002", "P003", "P001"],
        "provider_id":     ["PR100", "PR101", "PR102", "PR100"],
        "diagnosis_code":  ["D10", None, "D30", "D10"],
        "procedure_code":  ["PROC1", "PROC2", None, "PROC1"],
        "billed_amount":   [5000.0, 50000.0, None, 5000.0],
        "date":            ["2024-01-01", "2024-02-15", "2024-03-10", "2024-01-01"],
        "ingestion_timestamp": ["ts"] * 4,
        "source_file":         ["claims.csv"] * 4,
    })


@pytest.fixture()
def clean_claims_df(raw_claims: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaner to the raw claims fixture."""
    return clean_claims(raw_claims)


@pytest.fixture()
def providers() -> pd.DataFrame:
    return pd.DataFrame({
        "provider_id":  ["PR100", "PR101", "PR102"],
        "doctor_name":  ["Dr A", "Dr B", "Dr C"],
        "specialty":    ["Cardiology", "Neurology", "General"],
        "location":     ["Delhi", None, "Mumbai"],   # PR101 has missing location
    })


@pytest.fixture()
def diagnosis() -> pd.DataFrame:
    return pd.DataFrame({
        "diagnosis_code": ["D10", "D20", "D30", "D40", "D50", "D60"],
        "category":       ["Heart", "Bone", "Fever", "Skin", "Diabetes", "Cold"],
        "severity":       ["High", "High", "Low", "Low", "High", "Low"],
    })


@pytest.fixture()
def cost() -> pd.DataFrame:
    return pd.DataFrame({
        "procedure_code": ["PROC1", "PROC2"],
        "average_cost":   [4000.0, 12000.0],
        "expected_cost":  [5000.0, 15000.0],
    })


# --------------------------------------------------------------------------- #
#  cleaner.py tests                                                             #
# --------------------------------------------------------------------------- #

class TestCleanClaims:
    def test_removes_duplicate_claim_ids(self, clean_claims_df: pd.DataFrame) -> None:
        """Duplicate claim_ids must be dropped (keep first occurrence)."""
        assert clean_claims_df["claim_id"].duplicated().sum() == 0

    def test_row_count_after_dedup(self, clean_claims_df: pd.DataFrame) -> None:
        """3 unique claims remain after removing the 1 duplicate."""
        assert len(clean_claims_df) == 3

    def test_date_parsed_to_datetime(self, clean_claims_df: pd.DataFrame) -> None:
        """'date' column must be converted to a datetime 'claim_date' column."""
        assert "claim_date" in clean_claims_df.columns
        assert "date" not in clean_claims_df.columns
        assert pd.api.types.is_datetime64_any_dtype(clean_claims_df["claim_date"])

    def test_diagnosis_code_uppercased(self, raw_claims: pd.DataFrame) -> None:
        """Diagnosis codes must be uppercased regardless of input casing."""
        raw_claims.loc[0, "diagnosis_code"] = "d10"  # lowercase input
        cleaned = clean_claims(raw_claims.drop_duplicates(subset=["claim_id"]))
        assert cleaned.loc[cleaned["claim_id"] == "C001", "diagnosis_code"].iloc[0] == "D10"


class TestEnrichClaims:
    def test_specialty_joined(
        self,
        clean_claims_df: pd.DataFrame,
        providers: pd.DataFrame,
        diagnosis: pd.DataFrame,
        cost: pd.DataFrame,
    ) -> None:
        """Provider specialty must appear after enrichment join."""
        from src.silver.cleaner import clean_providers, clean_diagnosis, clean_cost
        enriched = enrich_claims(
            clean_claims_df,
            clean_providers(providers),
            clean_diagnosis(diagnosis),
            clean_cost(cost),
        )
        assert "specialty" in enriched.columns

    def test_expected_cost_joined(
        self,
        clean_claims_df: pd.DataFrame,
        providers: pd.DataFrame,
        diagnosis: pd.DataFrame,
        cost: pd.DataFrame,
    ) -> None:
        """expected_cost must appear after cost reference join."""
        from src.silver.cleaner import clean_providers, clean_diagnosis, clean_cost
        enriched = enrich_claims(
            clean_claims_df,
            clean_providers(providers),
            clean_diagnosis(diagnosis),
            clean_cost(cost),
        )
        assert "expected_cost" in enriched.columns

    def test_all_claims_preserved(
        self,
        clean_claims_df: pd.DataFrame,
        providers: pd.DataFrame,
        diagnosis: pd.DataFrame,
        cost: pd.DataFrame,
    ) -> None:
        """Left join must preserve all claim rows even with missing reference codes."""
        from src.silver.cleaner import clean_providers, clean_diagnosis, clean_cost
        enriched = enrich_claims(
            clean_claims_df,
            clean_providers(providers),
            clean_diagnosis(diagnosis),
            clean_cost(cost),
        )
        assert len(enriched) == len(clean_claims_df)


# --------------------------------------------------------------------------- #
#  validator.py tests                                                           #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def enriched_df(
    clean_claims_df, providers, diagnosis, cost
) -> pd.DataFrame:
    """Build a minimal enriched Silver DataFrame for validation tests."""
    from src.silver.cleaner import clean_providers, clean_diagnosis, clean_cost
    df = enrich_claims(
        clean_claims_df,
        clean_providers(providers),
        clean_diagnosis(diagnosis),
        clean_cost(cost),
    )
    # Add validation_flags placeholder so the fixture is self-contained
    df["validation_flags"] = ""
    return df


class TestValidator:
    def test_missing_diagnosis_flag(self, enriched_df: pd.DataFrame) -> None:
        """flag_missing_diagnosis must be True for rows with no diagnosis_code."""
        validated = apply_validation_rules(enriched_df)
        missing_mask = enriched_df["diagnosis_code"].isna()
        assert validated.loc[missing_mask, "flag_missing_diagnosis"].all()

    def test_missing_procedure_flag(self, enriched_df: pd.DataFrame) -> None:
        """flag_missing_procedure must be True for rows with no procedure_code."""
        validated = apply_validation_rules(enriched_df)
        missing_mask = enriched_df["procedure_code"].isna()
        assert validated.loc[missing_mask, "flag_missing_procedure"].all()

    def test_high_billing_flag(self) -> None:
        """flag_high_billing fires when billed > 3× expected_cost."""
        df = pd.DataFrame({
            "claim_id":        ["C999"],
            "diagnosis_code":  ["D10"],
            "procedure_code":  ["PROC1"],
            "billed_amount":   [50000.0],   # 50k >> 3× 5k threshold
            "expected_cost":   [5000.0],
            "validation_flags": [""],
        })
        validated = apply_validation_rules(df)
        assert validated.loc[0, "flag_high_billing"] is True or \
               validated.loc[0, "flag_high_billing"] == True

    def test_incomplete_claim_flag(self, enriched_df: pd.DataFrame) -> None:
        """flag_incomplete_claim fires only when BOTH codes are missing."""
        validated = apply_validation_rules(enriched_df)
        both_missing = (
            enriched_df["diagnosis_code"].isna()
            & enriched_df["procedure_code"].isna()
        )
        assert validated.loc[both_missing, "flag_incomplete_claim"].all()

    def test_validation_flags_string(self, enriched_df: pd.DataFrame) -> None:
        """validation_flags must be a non-null string column.

        NOTE: pandas 3 uses StringDtype instead of object for string columns.
        We use pd.api.types.is_string_dtype() which returns True for both
        legacy object dtype (pandas 2) and the new StringDtype (pandas 3).
        """
        validated = apply_validation_rules(enriched_df)
        assert "validation_flags" in validated.columns
        assert pd.api.types.is_string_dtype(validated["validation_flags"])
        assert validated["validation_flags"].notna().all()


    def test_exception_hierarchy(self) -> None:
        """IngestionError and SilverPipelineError must be ClaimDenialSystemError subclasses."""
        from src.core.exceptions import ClaimDenialSystemError
        ing_err = IngestionError(error_code="ING-1001", message="test")
        slv_err = SilverPipelineError(error_code="SLV-2001", message="test")
        assert isinstance(ing_err, ClaimDenialSystemError)
        assert isinstance(slv_err, ClaimDenialSystemError)
        # Verify the error_code is stored and accessible
        assert ing_err.error_code == "ING-1001"
        assert slv_err.error_code == "SLV-2001"

    def test_exception_str_format(self) -> None:
        """str(exception) must include the error code for searchable log lines."""
        ing_err = IngestionError(error_code="ING-1001", message="file not found")
        assert "ING-1001" in str(ing_err)
        assert "file not found" in str(ing_err)


    def test_clean_claim_has_empty_flags(self) -> None:
        """A fully populated valid claim must have no flags."""
        df = pd.DataFrame({
            "claim_id":        ["C888"],
            "diagnosis_code":  ["D10"],
            "procedure_code":  ["PROC1"],
            "billed_amount":   [4000.0],
            "expected_cost":   [5000.0],
            "validation_flags": [""],
        })
        validated = apply_validation_rules(df)
        assert validated.loc[0, "validation_flags"] == ""
