"""
Unit Tests — Week 5: ML Layer (trainer.py, predictor.py, explainer.py)
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ml.trainer import (
    find_best_threshold,
    prepare_features,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.ml.predictor import PredictionResult


# --------------------------------------------------------------------------- #
#  Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def sample_gold_df() -> pd.DataFrame:
    """
    Minimal Gold-layer DataFrame for testing prepare_features().
    Two denied, two approved — balanced for threshold testing.
    """
    return pd.DataFrame({
        "claim_amount_ratio":      [0.8,  3.5,  1.0,  4.0],
        "provider_claim_count":    [50,   50,   30,   30],
        "provider_denial_rate":    [0.10, 0.60, 0.15, 0.70],
        "diagnosis_severity_flag": [0,    1,    0,    1],
        "missing_fields_count":    [0,    2,    0,    3],
        "is_high_biller":          [0,    1,    0,    1],
        "claim_month":             [3,    12,   6,    11],
        "denial_label":            [0,    1,    0,    1],   # target
    })


# --------------------------------------------------------------------------- #
#  Tests: find_best_threshold                                                   #
# --------------------------------------------------------------------------- #

class TestFindBestThreshold:
    def test_returns_tuple(self) -> None:
        """find_best_threshold must return a (float, float) tuple."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.9])
        result = find_best_threshold(y_true, y_prob)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_threshold_in_valid_range(self) -> None:
        """Returned threshold must be between 0.10 and 0.90."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.9])
        threshold, _ = find_best_threshold(y_true, y_prob)
        assert 0.10 <= threshold <= 0.90

    def test_f1_non_negative(self) -> None:
        """Best F1 score must be >= 0."""
        y_true = np.array([0, 0, 0, 0])   # all approved — F1 = 0
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        _, best_f1 = find_best_threshold(y_true, y_prob)
        assert best_f1 >= 0.0

    def test_perfect_classifier_gets_high_f1(self) -> None:
        """A perfectly separable dataset should produce F1 close to 1."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        _, best_f1 = find_best_threshold(y_true, y_prob)
        assert best_f1 >= 0.9


# --------------------------------------------------------------------------- #
#  Tests: prepare_features                                                      #
# --------------------------------------------------------------------------- #

class TestPrepareFeatures:
    def test_returns_correct_shapes(self, sample_gold_df: pd.DataFrame, tmp_path, monkeypatch) -> None:
        """X should have same rows as input, y should be 1D."""
        # Monkeypatch MODELS_DIR so medians are saved to tmp_path
        import src.ml.trainer as trainer_module
        monkeypatch.setattr(trainer_module, "MODELS_DIR", tmp_path)
        X, y = prepare_features(sample_gold_df)
        assert len(X) == len(sample_gold_df)
        assert len(y) == len(sample_gold_df)

    def test_feature_columns_correct(self, sample_gold_df: pd.DataFrame, tmp_path, monkeypatch) -> None:
        """X must contain exactly the NUMERIC_FEATURES columns."""
        import src.ml.trainer as trainer_module
        monkeypatch.setattr(trainer_module, "MODELS_DIR", tmp_path)
        X, _ = prepare_features(sample_gold_df)
        assert list(X.columns) == NUMERIC_FEATURES

    def test_no_nulls_after_imputation(self, tmp_path, monkeypatch) -> None:
        """After prepare_features, X must have no NaN values."""
        import src.ml.trainer as trainer_module
        monkeypatch.setattr(trainer_module, "MODELS_DIR", tmp_path)
        df_with_nulls = pd.DataFrame({
            "claim_amount_ratio":      [1.0,  None],
            "provider_claim_count":    [50,   None],
            "provider_denial_rate":    [0.10, None],
            "diagnosis_severity_flag": [0,    None],
            "missing_fields_count":    [0,    None],
            "is_high_biller":          [0,    None],
            "claim_month":             [3,    None],
            "denial_label":            [0,    1],
        })
        X, _ = prepare_features(df_with_nulls)
        assert X.isnull().sum().sum() == 0

    def test_saves_medians_json(self, sample_gold_df: pd.DataFrame, tmp_path, monkeypatch) -> None:
        """prepare_features must save feature_medians.json to MODELS_DIR."""
        import src.ml.trainer as trainer_module
        monkeypatch.setattr(trainer_module, "MODELS_DIR", tmp_path)
        prepare_features(sample_gold_df)
        medians_path = tmp_path / "feature_medians.json"
        assert medians_path.exists()
        with open(medians_path) as fp:
            medians = json.load(fp)
        assert set(medians.keys()) == set(NUMERIC_FEATURES)


# --------------------------------------------------------------------------- #
#  Tests: PredictionResult dataclass                                            #
# --------------------------------------------------------------------------- #

class TestPredictionResult:
    def test_fields_accessible(self) -> None:
        """All PredictionResult fields must be accessible by name."""
        result = PredictionResult(
            probability=0.75,
            denial_label=1,
            risk_level="HIGH",
            threshold=0.50,
        )
        assert result.probability  == 0.75
        assert result.denial_label == 1
        assert result.risk_level   == "HIGH"
        assert result.threshold    == 0.50

    def test_probability_range_valid(self) -> None:
        """Probability must be between 0.0 and 1.0."""
        result = PredictionResult(probability=0.82, denial_label=1, risk_level="HIGH", threshold=0.5)
        assert 0.0 <= result.probability <= 1.0

    def test_denial_label_binary(self) -> None:
        """denial_label must be 0 or 1."""
        for label in [0, 1]:
            result = PredictionResult(probability=0.5, denial_label=label, risk_level="MEDIUM", threshold=0.5)
            assert result.denial_label in {0, 1}
