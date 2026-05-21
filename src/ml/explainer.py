"""
src/ml/explainer.py
-------------------
Week 5 — SHAP-based explainability for the denial prediction model.

WHY SHAP?
---------
SHAP (SHapley Additive exPlanations) answers the question:
"WHICH features most contributed to THIS specific claim's risk score?"

Unlike global feature importance (which tells you what the model cares about
on average), SHAP values are per-claim. So for Claim C001 we can say:
    "billed amount was 4× expected → +0.35 risk increase"
    "provider has 40% historical denial rate → +0.22 risk increase"
    "diagnosis code is present → -0.12 risk reduction"

This is what the dashboard and remediation agent show the billing analyst.

HOW IT WORKS
------------
We use SHAP's TreeExplainer, which is optimised for XGBoost and runs in
milliseconds (no GPU, no heavy computation). It outputs a SHAP value per
feature per prediction. Positive = pushed score up (toward denial).
Negative = pushed score down (toward approval).

Usage
-----
    from src.ml.explainer import explain

    attributions = explain({
        "claim_amount_ratio":      3.8,
        "provider_claim_count":    45,
        "provider_denial_rate":    0.62,
        "diagnosis_severity_flag": 1,
        "missing_fields_count":    2,
        "is_high_biller":          1,
        "claim_month":             12,
    })

    for attr in attributions:
        print(attr.feature_label, attr.direction, attr.shap_value)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

from src.core.logger import get_logger
from src.core.error_codes import ErrorCode
from src.core.exceptions import PredictionError
# Use the PUBLIC accessor functions — never import private module-level variables
# from another module. See predictor.py get_model() / get_feature_medians() docstrings.
from src.ml.predictor import FEATURE_COLUMNS, _load_artifacts, get_model, get_feature_medians

# --------------------------------------------------------------------------- #
#  Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Human-readable labels for each feature                                       #
# --------------------------------------------------------------------------- #
# Maps internal feature column name → label shown to the billing analyst.
# All labels are actionable — the analyst should be able to read these and
# know what to fix.
FEATURE_LABELS: dict[str, str] = {
    "claim_amount_ratio":      "Billed vs Expected Cost Ratio",
    "provider_claim_count":    "Provider Volume (total claims)",
    "provider_denial_rate":    "Provider Historical Denial Rate",
    "diagnosis_severity_flag": "Diagnosis Severity (High = 1)",
    "missing_fields_count":    "Number of Missing Required Fields",
    "is_high_biller":          "High Billing Flag (billed > 2× expected)",
    "claim_month":             "Claim Month (seasonal pattern)",
}

# Cached SHAP explainer — initialised on first call
_explainer: Optional[shap.TreeExplainer] = None


# --------------------------------------------------------------------------- #
#  Return type                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class FeatureAttribution:
    """
    SHAP attribution for a single feature on a single claim.

    Why a dataclass?
    ----------------
    Typed output makes it trivial for the dashboard and agent to iterate
    over attributions and render them consistently without parsing dicts.

    Attributes:
        feature_name:  Internal column name (e.g. "claim_amount_ratio").
        feature_label: Human-readable label shown in the UI.
        feature_value: The actual value of this feature for this claim.
        shap_value:    How much this feature contributed to the risk score.
                       Positive = pushed toward denial, negative = toward approval.
        direction:     "INCREASES_RISK" or "REDUCES_RISK" — derived from shap_value sign.
    """
    feature_name:  str    # internal name, matches FEATURE_COLUMNS
    feature_label: str    # label shown in UI
    feature_value: float  # actual value for this claim
    shap_value:    float  # SHAP contribution (positive = toward denial)
    direction:     str    # "INCREASES_RISK" | "REDUCES_RISK"


# --------------------------------------------------------------------------- #
#  Explainer loader                                                             #
# --------------------------------------------------------------------------- #

def _load_explainer() -> None:
    """
    Initialise the SHAP TreeExplainer using the loaded XGBoost model.

    The explainer is cached in _explainer after the first call.

    WHY underscore prefix?
    This is an internal initialisation helper. External code calls explain()
    which handles lazy-loading automatically.

    Raises:
        ModelNotFoundError: If the model has not been trained yet.
        PredictionError: If the TreeExplainer fails to initialise.
    """
    global _explainer

    # Use the PUBLIC get_model() accessor — this ensures the model is loaded
    # and cached in predictor.py regardless of import order.
    model = get_model()

    try:
        # TreeExplainer is the correct SHAP explainer for XGBoost models.
        # It runs in O(n_trees) time — very fast compared to KernelExplainer.
        _explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialised.")
    except Exception as exc:
        logger.error(
            "[%s] Failed to initialise SHAP explainer. error=%s",
            ErrorCode.ML_SHAP_FAILED, str(exc),
        )
        raise PredictionError(
            error_code=ErrorCode.ML_SHAP_FAILED,
            message=f"SHAP TreeExplainer failed to initialise: {exc}",
        ) from exc


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

def explain(features: dict) -> list[FeatureAttribution]:
    """
    Compute SHAP attributions for a single claim.

    Args:
        features: Same feature dict format as predictor.predict().
                  Missing features are filled with training medians.

    Returns:
        List of FeatureAttribution, sorted by absolute SHAP value descending.
        First item = most impactful feature for this claim.

    Raises:
        RuntimeError: If SHAP computation fails.
    """
    # Load explainer on first call
    if _explainer is None:
        _load_explainer()

    # Re-use the same feature preparation logic as predictor.predict():
    # fill missing features with training medians, build a one-row DataFrame.
    # Use the PUBLIC get_feature_medians() accessor — not _feature_medians directly.
    feature_medians = get_feature_medians()
    row: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        if col in features and features[col] is not None:
            row[col] = float(features[col])
        else:
            row[col] = float(feature_medians.get(col, 0.0))

    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    try:
        # Compute SHAP values — returns array of shape (1, n_features)
        shap_values: np.ndarray = _explainer.shap_values(X)   # type: ignore[union-attr]

        # For binary XGBoost, shap_values is shape (1, n_features).
        # Each value is the log-odds contribution of that feature.
        shap_row: np.ndarray = shap_values[0]

    except Exception as exc:
        logger.error(
            "[%s] SHAP computation failed. error=%s",
            ErrorCode.ML_SHAP_FAILED, str(exc),
        )
        raise PredictionError(
            error_code=ErrorCode.ML_SHAP_FAILED,
            message=f"SHAP computation failed: {exc}",
        ) from exc

    # Build FeatureAttribution objects
    attributions: list[FeatureAttribution] = []
    for i, col in enumerate(FEATURE_COLUMNS):
        shap_val     = float(shap_row[i])
        direction    = "INCREASES_RISK" if shap_val > 0 else "REDUCES_RISK"
        feature_val  = row[col]
        label        = FEATURE_LABELS.get(col, col)

        attributions.append(FeatureAttribution(
            feature_name=col,
            feature_label=label,
            feature_value=round(feature_val, 4),
            shap_value=round(shap_val, 4),
            direction=direction,
        ))

    # Sort by absolute impact — most influential feature first
    attributions.sort(key=lambda a: abs(a.shap_value), reverse=True)

    logger.debug(
        "SHAP explanation complete. features=%d top_feature=%s top_shap=%.4f",
        len(attributions),
        attributions[0].feature_name if attributions else "N/A",
        attributions[0].shap_value   if attributions else 0.0,
    )

    return attributions


def format_explanation_text(attributions: list[FeatureAttribution]) -> list[str]:
    """
    Convert SHAP attributions into human-readable sentences for the UI.

    Each sentence is actionable — the billing analyst should be able to read
    it and know exactly what to fix before resubmitting the claim.

    Args:
        attributions: Output of explain(), sorted by impact.

    Returns:
        List of plain-English explanation strings, one per feature.

    Example output:
        ["⚠ Billed vs Expected Cost Ratio is 3.8 — significantly above benchmark (+0.35 risk)",
         "⚠ Number of Missing Required Fields is 2 — incomplete claim increases denial risk (+0.22 risk)",
         "✓ Diagnosis Severity (High = 1) is 1 — slightly reduces risk (-0.08 risk)"]
    """
    lines: list[str] = []

    for attr in attributions:
        # Only show features with non-trivial SHAP impact (threshold: 0.01)
        if abs(attr.shap_value) < 0.01:
            continue

        # Choose icon and direction text based on whether feature hurts or helps
        if attr.direction == "INCREASES_RISK":
            icon      = "⚠"
            impact    = f"+{attr.shap_value:.2f} risk"
        else:
            icon      = "✓"
            impact    = f"{attr.shap_value:.2f} risk"

        line = (
            f"{icon} {attr.feature_label} is {attr.feature_value} "
            f"({impact})"
        )
        lines.append(line)

    return lines
