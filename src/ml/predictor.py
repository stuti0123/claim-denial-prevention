"""
src/ml/predictor.py
-------------------
Week 5 — Load the saved model and predict denial risk for a single claim.

WHAT THIS FILE DOES
-------------------
At import time (or first call), loads three artifacts from models/:
    - denial_model.pkl      — the trained XGBoost model
    - threshold.json        — the decision threshold chosen by trainer.py
    - feature_medians.json  — median values for imputing missing features

At prediction time:
    1. Receives a flat dict of feature values (from API or agent).
    2. Fills any missing features with the saved medians.
    3. Runs model.predict_proba() to get denial probability.
    4. Applies threshold to get binary label.
    5. Maps probability to a human-readable risk level.

RISK LEVELS
-----------
    HIGH   — probability >= 0.70   (flag immediately, high confidence denial)
    MEDIUM — probability 0.40–0.69 (review recommended)
    LOW    — probability < 0.40    (likely to be approved)

Usage
-----
    from src.ml.predictor import predict

    result = predict({
        "claim_amount_ratio":      2.5,
        "provider_claim_count":    120,
        "provider_denial_rate":    0.45,
        "diagnosis_severity_flag": 1,
        "missing_fields_count":    1,
        "is_high_biller":          0,
        "claim_month":             3,
    })
    # result = PredictionResult(probability=0.81, denial_label=1, risk_level="HIGH")
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.core.logger import get_logger
from src.core.error_codes import ErrorCode
from src.core.exceptions import ModelNotFoundError, PredictionError

# --------------------------------------------------------------------------- #
#  Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Feature list — MUST match trainer.py exactly                                 #
# --------------------------------------------------------------------------- #
# This list defines the column order that XGBoost expects.
# Any change here must also be reflected in trainer.py → NUMERIC_FEATURES.
FEATURE_COLUMNS: list[str] = [
    "claim_amount_ratio",
    "provider_claim_count",
    "provider_denial_rate",
    "diagnosis_severity_flag",
    "missing_fields_count",
    "is_high_biller",
    "claim_month",
]

# Risk level thresholds — tuned for the healthcare billing context.
# HIGH means the system is highly confident the claim will be denied.
RISK_HIGH_THRESHOLD:   float = 0.70
RISK_MEDIUM_THRESHOLD: float = 0.40


# --------------------------------------------------------------------------- #
#  Return type                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class PredictionResult:
    """
    Structured output from a single claim prediction.

    Why a dataclass?
    ----------------
    Using a dataclass makes the contract explicit — the API and agent both
    know exactly what fields to expect. It also gives us free __repr__ and
    type hints, making debugging much easier than returning a plain dict.

    Attributes:
        probability:  Model-estimated probability of denial (0.0–1.0).
        denial_label: Binary prediction — 1 = denied, 0 = approved.
        risk_level:   Human-readable risk category ("HIGH", "MEDIUM", "LOW").
        threshold:    The decision threshold used (from threshold.json).
    """
    probability:  float   # 0.0 (definitely approved) to 1.0 (definitely denied)
    denial_label: int     # 0 = approved, 1 = denied
    risk_level:   str     # "HIGH", "MEDIUM", or "LOW"
    threshold:    float   # the threshold value used for this prediction


# --------------------------------------------------------------------------- #
#  Model cache — loaded once per process                                        #
# --------------------------------------------------------------------------- #

# These variables hold the loaded artifacts in memory.
# They are None until _load_artifacts() is called on first predict().
_model:           Optional[object]          = None
_threshold:       Optional[float]           = None
_feature_medians: Optional[dict[str, float]] = None


def _load_artifacts() -> None:
    """
    Load the model, threshold, and feature medians from disk.

    Called automatically on the first call to predict(). Subsequent calls
    use the cached objects — loading happens only once per process.

    WHY underscore prefix (_load_artifacts)?
    This is an internal initialisation function. External code should call
    predict() which handles lazy-loading automatically. Callers should never
    need to call this directly.

    Raises:
        ModelNotFoundError: If any required artifact file is missing or corrupt.
    """
    global _model, _threshold, _feature_medians

    # Load the trained XGBoost model
    model_path = MODELS_DIR / "denial_model.pkl"
    if not model_path.exists():
        logger.error(
            "[%s] Model file not found. path=%s",
            ErrorCode.ML_MODEL_NOT_FOUND, model_path,
        )
        raise ModelNotFoundError(
            error_code=ErrorCode.ML_MODEL_NOT_FOUND,
            message=f"Model not found at {model_path}. Run: python -m src.ml.trainer",
        )
    try:
        _model = joblib.load(model_path)
    except Exception as exc:
        logger.error(
            "[%s] Failed to load model file. path=%s error=%s",
            ErrorCode.ML_MODEL_NOT_FOUND, model_path, str(exc),
        )
        raise ModelNotFoundError(
            error_code=ErrorCode.ML_MODEL_NOT_FOUND,
            message=f"Failed to load model from {model_path}: {exc}",
        ) from exc
    logger.info("Model loaded. path=%s", model_path)

    # Load the decision threshold chosen by the trainer
    threshold_path = MODELS_DIR / "threshold.json"
    if not threshold_path.exists():
        logger.error(
            "[%s] Threshold file not found. path=%s",
            ErrorCode.ML_MODEL_NOT_FOUND, threshold_path,
        )
        raise ModelNotFoundError(
            error_code=ErrorCode.ML_MODEL_NOT_FOUND,
            message=f"Threshold file not found: {threshold_path}",
        )
    try:
        with open(threshold_path, "r", encoding="utf-8") as fp:
            threshold_data = json.load(fp)
        _threshold = threshold_data["threshold"]
    except json.JSONDecodeError as exc:
        # Catches corrupt/truncated JSON files — not caught by FileNotFoundError
        logger.error(
            "[%s] Threshold file is corrupt (invalid JSON). path=%s error=%s",
            ErrorCode.ML_MODEL_NOT_FOUND, threshold_path, str(exc),
        )
        raise ModelNotFoundError(
            error_code=ErrorCode.ML_MODEL_NOT_FOUND,
            message=f"threshold.json is corrupt: {exc}",
        ) from exc
    except Exception as exc:
        logger.error(
            "[%s] Failed to read threshold file. path=%s error=%s",
            ErrorCode.ML_MODEL_NOT_FOUND, threshold_path, str(exc),
        )
        raise ModelNotFoundError(
            error_code=ErrorCode.ML_MODEL_NOT_FOUND,
            message=f"Failed to read threshold.json: {exc}",
        ) from exc
    logger.info("Threshold loaded. value=%.2f", _threshold)

    # Load feature medians for imputing missing values at inference time
    medians_path = MODELS_DIR / "feature_medians.json"
    if not medians_path.exists():
        logger.error(
            "[%s] Feature medians file not found. path=%s",
            ErrorCode.ML_MODEL_NOT_FOUND, medians_path,
        )
        raise ModelNotFoundError(
            error_code=ErrorCode.ML_MODEL_NOT_FOUND,
            message=f"Feature medians not found: {medians_path}",
        )
    try:
        with open(medians_path, "r", encoding="utf-8") as fp:
            _feature_medians = json.load(fp)
    except json.JSONDecodeError as exc:
        logger.error(
            "[%s] Feature medians file is corrupt (invalid JSON). path=%s error=%s",
            ErrorCode.ML_MODEL_NOT_FOUND, medians_path, str(exc),
        )
        raise ModelNotFoundError(
            error_code=ErrorCode.ML_MODEL_NOT_FOUND,
            message=f"feature_medians.json is corrupt: {exc}",
        ) from exc
    except Exception as exc:
        logger.error(
            "[%s] Failed to read feature medians. path=%s error=%s",
            ErrorCode.ML_MODEL_NOT_FOUND, medians_path, str(exc),
        )
        raise ModelNotFoundError(
            error_code=ErrorCode.ML_MODEL_NOT_FOUND,
            message=f"Failed to read feature_medians.json: {exc}",
        ) from exc
    logger.info("Feature medians loaded. features=%d", len(_feature_medians))

# --------------------------------------------------------------------------- #
#  Public artifact accessor (used by explainer.py)                              #
# --------------------------------------------------------------------------- #

def get_model() -> object:
    """
    Return the loaded XGBoost model, loading it if necessary.

    WHY A PUBLIC FUNCTION instead of importing _model directly?
    Explainer.py originally imported _model (a private module-level variable)
    directly. This created hidden state coupling: if predictor hadn't run yet,
    _model would be None in explainer. A public function with lazy-loading
    ensures the model is always ready, regardless of import order.

    Returns:
        The loaded XGBoost model object.

    Raises:
        ModelNotFoundError: If the model artifact is missing.
    """
    if _model is None:
        _load_artifacts()
    return _model


def get_feature_medians() -> dict[str, float]:
    """
    Return the feature medians dict, loading artifacts if necessary.

    Used by explainer.py to impute missing features consistently with
    how the predictor imputes them — training and inference must match.

    Returns:
        Dict mapping feature name to its training-time median.

    Raises:
        ModelNotFoundError: If the medians artifact is missing.
    """
    if _feature_medians is None:
        _load_artifacts()
    assert _feature_medians is not None
    return _feature_medians


def predict(features: dict) -> PredictionResult:
    """
    Predict denial risk for a single claim.

    Args:
        features: Dict mapping feature names to values. Missing features are
                  filled with the median computed during training.
                  Expected keys (all optional — missing = imputed):
                    claim_amount_ratio, provider_claim_count,
                    provider_denial_rate, diagnosis_severity_flag,
                    missing_fields_count, is_high_biller, claim_month

    Returns:
        PredictionResult with probability, denial_label, risk_level, threshold.

    Raises:
        ModelNotFoundError: If model artifacts are not found (not trained yet).
        PredictionError: If model inference fails.
    """
    # Load artifacts on first call — cached for subsequent calls
    if _model is None:
        _load_artifacts()

    # Build a one-row DataFrame in the exact column order the model expects.
    # Missing features are filled with their training-time medians.
    row: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        if col in features and features[col] is not None:
            row[col] = float(features[col])
        else:
            # Fill missing feature with training median
            median_val = _feature_medians.get(col, 0.0)   # type: ignore[union-attr]
            row[col]   = float(median_val)
            logger.debug(
                "Feature missing at inference — imputing median. feature=%s median=%.4f",
                col, median_val,
            )

    # Create a single-row DataFrame for the model
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    try:
        # I/O boundary: model inference can fail if feature schema doesn't match
        probability: float = float(_model.predict_proba(X)[0, 1])   # type: ignore[union-attr]
    except Exception as exc:
        logger.error(
            "[%s] Model inference failed. error=%s",
            ErrorCode.ML_FEATURE_MISMATCH, str(exc),
        )
        raise PredictionError(
            error_code=ErrorCode.ML_FEATURE_MISMATCH,
            message=f"Model inference failed: {exc}",
        ) from exc

    # Validate probability is in [0, 1] — should never be out of range for XGBoost
    # but we guard defensively in case of future model changes.
    if not (0.0 <= probability <= 1.0):
        logger.error(
            "[%s] Model returned invalid probability. value=%.4f",
            ErrorCode.ML_INVALID_PROBABILITY, probability,
        )
        probability = max(0.0, min(1.0, probability))   # clamp to safe range

    # Apply threshold to get binary label
    denial_label: int = int(probability >= _threshold)   # type: ignore[arg-type]

    # Map probability to human-readable risk level
    if probability >= RISK_HIGH_THRESHOLD:
        risk_level = "HIGH"
    elif probability >= RISK_MEDIUM_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    logger.debug(
        "Prediction complete. probability=%.4f label=%d risk=%s threshold=%.2f",
        probability, denial_label, risk_level, _threshold,
    )

    return PredictionResult(
        probability=round(probability, 4),
        denial_label=denial_label,
        risk_level=risk_level,
        threshold=_threshold,   # type: ignore[arg-type]
    )
