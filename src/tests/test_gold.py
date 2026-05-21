"""
Unit Tests — Week 4: Gold Layer (feature_engineer.py)
"""

import numpy as np
import pandas as pd
import pytest

from src.gold.feature_engineer import (
    add_claim_amount_ratio,
    add_diagnosis_severity_flag,
    add_is_high_biller,
    add_missing_fields_count,
    add_claim_month,
    add_provider_features,
    add_synthetic_denial_label,
    DENIAL_THRESHOLD,
)


# --------------------------------------------------------------------------- #
#  Fixture — minimal validated Silver DataFrame                                 #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def base_df() -> pd.DataFrame:
    """
    Minimal validated Silver DataFrame covering several risk combinations.
    Row 0: clean claim (should be approved)
    Row 1: missing diagnosis only
    Row 2: missing both diagnosis and procedure (ERR_INCOMPLETE_CLAIM)
    Row 3: high billing
    """
    return pd.DataFrame({
        "claim_id":              ["C001", "C002", "C003", "C004"],
        "patient_id":            ["P001", "P002", "P003", "P004"],
        "provider_id":           ["PR100", "PR100", "PR101", "PR102"],
        "diagnosis_code":        ["D10",   None,   None,   "D50"],
        "procedure_code":        ["PROC1", "PROC2", None,  "PROC2"],
        "billed_amount":         [4000.0,  10000.0, None,  50000.0],
        "expected_cost":         [5000.0,  15000.0, 5000.0, 5000.0],
        "severity":              ["High",  None,    None,  "High"],
        "claim_date":            pd.to_datetime(
            ["2024-01-15", "2024-02-20", "2024-03-05", "2024-06-10"]
        ),
        # Flags from validator (set manually for isolation)
        "flag_missing_diagnosis": [False, True,  True,  False],
        "flag_missing_procedure": [False, False, True,  False],
        "flag_missing_amount":    [False, False, True,  False],
        "flag_high_billing":      [False, False, False, True],
        "flag_invalid_diagnosis": [False, False, False, False],
        "flag_incomplete_claim":  [False, False, True,  False],
        "validation_flags":       ["", "WARN_MISSING_DIAGNOSIS",
                                   "WARN_MISSING_DIAGNOSIS|WARN_MISSING_PROCEDURE|WARN_MISSING_AMOUNT|ERR_INCOMPLETE_CLAIM",
                                   "WARN_HIGH_BILLING"],
    })


# --------------------------------------------------------------------------- #
#  Feature: claim_amount_ratio                                                  #
# --------------------------------------------------------------------------- #

class TestClaimAmountRatio:
    def test_column_exists(self, base_df: pd.DataFrame) -> None:
        result = add_claim_amount_ratio(base_df)
        assert "claim_amount_ratio" in result.columns

    def test_ratio_calculation(self, base_df: pd.DataFrame) -> None:
        """4000 / 5000 = 0.8 for the first clean claim."""
        result = add_claim_amount_ratio(base_df)
        assert abs(result.loc[0, "claim_amount_ratio"] - 0.8) < 1e-6

    def test_nan_when_expected_missing(self) -> None:
        """Ratio must be NaN when expected_cost is null."""
        df = pd.DataFrame({
            "billed_amount": [5000.0],
            "expected_cost": [None],
        })
        result = add_claim_amount_ratio(df)
        assert pd.isna(result.loc[0, "claim_amount_ratio"])

    def test_nan_when_billed_missing(self, base_df: pd.DataFrame) -> None:
        """Ratio must be NaN when billed_amount is null (Row 2)."""
        result = add_claim_amount_ratio(base_df)
        assert pd.isna(result.loc[2, "claim_amount_ratio"])


# --------------------------------------------------------------------------- #
#  Feature: diagnosis_severity_flag                                             #
# --------------------------------------------------------------------------- #

class TestDiagnosisSeverityFlag:
    def test_high_severity_is_1(self, base_df: pd.DataFrame) -> None:
        result = add_diagnosis_severity_flag(base_df)
        # Row 0 and Row 3 have severity='High'
        assert result.loc[0, "diagnosis_severity_flag"] == 1
        assert result.loc[3, "diagnosis_severity_flag"] == 1

    def test_missing_severity_is_0(self, base_df: pd.DataFrame) -> None:
        """Rows with no severity (joined null) should map to 0."""
        result = add_diagnosis_severity_flag(base_df)
        assert result.loc[1, "diagnosis_severity_flag"] == 0
        assert result.loc[2, "diagnosis_severity_flag"] == 0


# --------------------------------------------------------------------------- #
#  Feature: missing_fields_count                                                #
# --------------------------------------------------------------------------- #

class TestMissingFieldsCount:
    def test_zero_for_complete_claim(self, base_df: pd.DataFrame) -> None:
        result = add_missing_fields_count(base_df)
        assert result.loc[0, "missing_fields_count"] == 0

    def test_three_for_fully_incomplete(self, base_df: pd.DataFrame) -> None:
        """Row 2 is missing diagnosis, procedure, and billed_amount → count = 3."""
        result = add_missing_fields_count(base_df)
        assert result.loc[2, "missing_fields_count"] == 3


# --------------------------------------------------------------------------- #
#  Feature: is_high_biller                                                      #
# --------------------------------------------------------------------------- #

class TestIsHighBiller:
    def test_high_biller_flag(self, base_df: pd.DataFrame) -> None:
        """Row 3: 50000 > 2 × 5000 → is_high_biller = 1."""
        result = add_is_high_biller(base_df)
        assert result.loc[3, "is_high_biller"] == 1

    def test_not_high_biller(self, base_df: pd.DataFrame) -> None:
        """Row 0: 4000 < 2 × 5000 → is_high_biller = 0."""
        result = add_is_high_biller(base_df)
        assert result.loc[0, "is_high_biller"] == 0


# --------------------------------------------------------------------------- #
#  Feature: claim_month                                                         #
# --------------------------------------------------------------------------- #

class TestClaimMonth:
    def test_month_extracted(self, base_df: pd.DataFrame) -> None:
        result = add_claim_month(base_df)
        assert result.loc[0, "claim_month"] == 1   # January
        assert result.loc[3, "claim_month"] == 6   # June


# --------------------------------------------------------------------------- #
#  Synthetic denial label                                                       #
# --------------------------------------------------------------------------- #

class TestSyntheticDenialLabel:
    def test_columns_added(self, base_df: pd.DataFrame) -> None:
        result = add_synthetic_denial_label(base_df)
        assert "risk_score" in result.columns
        assert "denial_label" in result.columns

    def test_clean_claim_approved(self, base_df: pd.DataFrame) -> None:
        """Row 0 has no flags — risk_score = 0 → denial_label = 0."""
        result = add_synthetic_denial_label(base_df)
        assert result.loc[0, "risk_score"] == 0
        assert result.loc[0, "denial_label"] == 0

    def test_incomplete_claim_denied(self, base_df: pd.DataFrame) -> None:
        """
        Row 2: flag_incomplete_claim(+3) + flag_missing_diagnosis(+2)
               + flag_missing_procedure(+2) + flag_missing_amount(+1) = 8.
        8 >= DENIAL_THRESHOLD → denial_label = 1.
        """
        result = add_synthetic_denial_label(base_df)
        assert result.loc[2, "denial_label"] == 1

    def test_high_billing_denied(self, base_df: pd.DataFrame) -> None:
        """Row 3: flag_high_billing(+2) = 2. If threshold ≤ 2 → denied."""
        result = add_synthetic_denial_label(base_df)
        # Check the score is as expected
        assert result.loc[3, "risk_score"] == 2
        # Denial depends on DENIAL_THRESHOLD constant
        expected = 1 if 2 >= DENIAL_THRESHOLD else 0
        assert result.loc[3, "denial_label"] == expected

    def test_denial_label_is_binary(self, base_df: pd.DataFrame) -> None:
        """denial_label must only contain 0 or 1."""
        result = add_synthetic_denial_label(base_df)
        assert set(result["denial_label"].unique()).issubset({0, 1})
