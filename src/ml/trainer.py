"""
src/ml/trainer.py
-----------------
Week 5 — Train, evaluate, and save the XGBoost denial prediction model.

WHAT THIS FILE DOES
-------------------
1. Loads the Gold feature table (data/gold/gold_features.csv).
2. Prepares features — imputes medians for nulls, one-hot-encodes categoricals.
3. Splits into 80% train / 20% test, stratified on denial_label.
4. Trains an XGBoost binary classifier.
5. Selects the best probability threshold by maximising F1 on the test set.
   We scan all thresholds between 0.1–0.9 in steps of 0.01 and pick
   the one that gives the highest F1-score. This is better than defaulting to
   0.5 because our dataset has class imbalance (28% denied vs 72% approved).
6. Saves: the trained model, threshold, and feature importance to models/.
7. Logs all evaluation metrics so they appear in logs/app.log.

OUTPUT FILES
------------
    models/denial_model.pkl          — XGBoost model (joblib)
    models/threshold.json            — chosen decision threshold
    models/feature_importance.json   — feature names + importance scores
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
import optuna

from src.core.logger import get_logger
from src.core.error_codes import ErrorCode
from src.core.exceptions import GoldPipelineError, ModelNotFoundError

# --------------------------------------------------------------------------- #
#  Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR     = PROJECT_ROOT / "data" / "gold"
MODELS_DIR   = PROJECT_ROOT / "models"

# Create models directory if it does not exist yet
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
#  Feature configuration                                                        #
# --------------------------------------------------------------------------- #

# Numeric features fed to XGBoost.
# XGBoost can handle NaN natively — we still impute median to keep the
# predictor consistent with what the real-time API receives.
NUMERIC_FEATURES: list[str] = [
    "claim_amount_ratio",      # billed / expected — key overbilling signal
    "provider_claim_count",    # how active this provider is
    "provider_denial_rate",    # historical risk rate for this provider
    "diagnosis_severity_flag", # 1 = High severity diagnosis
    "missing_fields_count",    # 0–3 count of null critical fields
    "is_high_biller",          # 1 = billed > 2× expected
    "claim_month",             # 1–12 — captures seasonal patterns
]

# The column the model is trained to predict
TARGET_COLUMN: str = "denial_label"

# Train/test split ratio — 80% train, 20% test
TEST_SIZE:     float = 0.20

# Random seed for reproducibility across runs
RANDOM_STATE:  int   = 42


# --------------------------------------------------------------------------- #
#  Data preparation                                                             #
# --------------------------------------------------------------------------- #

def load_gold_data() -> pd.DataFrame:
    """
    Load the Gold feature table from disk.

    Returns:
        Gold DataFrame with all engineered features and the denial_label target.

    Raises:
        FileNotFoundError: If gold_features.csv does not exist.
                           Run src.gold.feature_engineer first.
    """
    gold_path = GOLD_DIR / "gold_features.csv"

    if not gold_path.exists():
        logger.error(
            "[%s] Gold features file not found. path=%s",
            ErrorCode.GLD_SILVER_MISSING, gold_path,
        )
        raise GoldPipelineError(
            error_code=ErrorCode.GLD_SILVER_MISSING,
            message=f"Gold features not found: {gold_path}. Run: python -m src.gold.feature_engineer",
        )

    try:
        df = pd.read_csv(gold_path)
    except Exception as exc:
        logger.error(
            "[%s] Failed to read Gold features file. path=%s error=%s",
            ErrorCode.GLD_SILVER_MISSING, gold_path, str(exc),
        )
        raise GoldPipelineError(
            error_code=ErrorCode.GLD_SILVER_MISSING,
            message=f"Failed to read Gold features file: {exc}",
        ) from exc

    logger.info("Gold data loaded. rows=%d columns=%d", len(df), len(df.columns))
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select feature columns and target, impute missing values.

    WHY MEDIAN IMPUTATION?
    ----------------------
    XGBoost handles NaN internally, but the real-time predictor receives a
    dict from the API that may be missing values. The predictor uses the same
    median values saved here, so training and inference are consistent.
    Args:
        df: Gold DataFrame loaded from gold_features.csv.

    Returns:
        Tuple of (X features DataFrame, y target Series).
    """
    # Select only the numeric features defined above
    X = df[NUMERIC_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    # Impute missing values with column medians — computed on the full dataset.
    # Medians are saved so the predictor can apply the same fill at inference.
    medians: dict[str, float] = X.median().to_dict()  # type: ignore[union-attr]  # Pylance: DataFrame.median() returns Series, not float
    X = X.fillna(medians)

    # Save medians so predictor.py can apply the same fill at inference time
    medians_path = MODELS_DIR / "feature_medians.json"
    try:
        with open(medians_path, "w", encoding="utf-8") as fp:
            json.dump(medians, fp, indent=2)
    except Exception as exc:
        logger.error(
            "[%s] Failed to save feature medians. path=%s error=%s",
            ErrorCode.GLD_WRITE_FAILED, medians_path, str(exc),
        )
        raise ModelNotFoundError(
            error_code=ErrorCode.GLD_WRITE_FAILED,
            message=f"Failed to write feature_medians.json: {exc}",
        ) from exc
    logger.info("Feature medians saved. path=%s", medians_path)

    # Log class distribution so we can spot imbalance
    denied_pct = y.mean() * 100
    logger.info(
        "Target distribution. denied=%.1f%% approved=%.1f%%",
        denied_pct, 100 - denied_pct,
    )

    return X, y  # type: ignore[return-value]  # X is DataFrame (df[list]), Pylance infers Series


# --------------------------------------------------------------------------- #
#  Threshold selection                                                          #
# --------------------------------------------------------------------------- #

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step: float = 0.01,
) -> tuple[float, float]:
    """
    Scan probability thresholds from 0.10 to 0.90 and return the one that
    maximises the F1-score on the provided labels.

    WHY NOT 0.5?
    ------------
    A default threshold of 0.5 assumes equal cost of false positives and
    false negatives. In healthcare billing, a missed denial (false negative)
    costs the hospital revenue while an over-flagged claim (false positive)
    is just extra review work. F1 balances both — but you could weight recall
    higher if the business cares more about catching denials.

    Args:
        y_true: True binary labels (0 = approved, 1 = denied).
        y_prob: Model-predicted probabilities for the denied class.
        step:   Threshold increment to scan (default 0.01 → 81 thresholds).

    Returns:
        Tuple of (best_threshold, best_f1_score).
    """
    best_threshold: float = 0.50
    best_f1:        float = 0.0

    # Scan thresholds from 0.10 to 0.90 inclusive
    for t in np.arange(0.10, 0.91, step):
        y_pred = (y_prob >= t).astype(int)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1       = f1
            best_threshold = float(round(t, 2))

    logger.info(
        "Threshold search complete. best_threshold=%.2f best_f1=%.4f",
        best_threshold, best_f1,
    )
    return best_threshold, best_f1


# --------------------------------------------------------------------------- #
#  Training pipeline                                                            #
# --------------------------------------------------------------------------- #

def train_model() -> dict:
    """
    Full training pipeline: load → prepare → train → evaluate → save.

    Steps:
      1. Load Gold data.
      2. Prepare feature matrix and target vector.
      3. Stratified 80/20 train/test split.
      4. Train XGBoost classifier.
      5. Find best threshold via F1 scan on train set.
      6. Log classification report and AUC-ROC on test set.
      7. Save model, threshold, and feature importance.

    Returns:
        Dict of evaluation metrics: auc, f1, threshold, model_path.

    Raises:
        GoldPipelineError: If Gold data cannot be loaded.
        ModelNotFoundError: If model artifacts cannot be saved.
    """
    logger.info("=== ML Training Pipeline Started ===")

    try:
        # Step 1 & 2: Load and prepare data
        df   = load_gold_data()
        X, y = prepare_features(df)

        # Step 3: Stratified split — preserves class ratio in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,           # ensures denial_label proportion is same in both splits
        )
        logger.info(
            "Data split. train=%d test=%d",
            len(X_train), len(X_test),
        )

        # Step 4: Hyperparameter Tuning with Optuna
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": scale_pos,
                "random_state": RANDOM_STATE,
                "eval_metric": "logloss",
                "verbosity": 0,
            }
            
            xgb = XGBClassifier(**params)
            
            # StratifiedKFold cross validation
            kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            auc_scores = []
            
            for train_idx, val_idx in kf.split(X_train, y_train):
                X_kf_train, X_kf_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_kf_train, y_kf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                xgb.fit(X_kf_train, y_kf_train)
                preds = xgb.predict_proba(X_kf_val)[:, 1]
                auc_scores.append(roc_auc_score(y_kf_val, preds))
                
            return np.mean(auc_scores)

        logger.info("Starting Optuna hyperparameter tuning...")
        # Suppress Optuna logging to avoid clutter
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10) # 10 trials for speed
        
        best_params = study.best_params
        best_params["scale_pos_weight"] = scale_pos
        best_params["random_state"] = RANDOM_STATE
        best_params["eval_metric"] = "logloss"
        best_params["verbosity"] = 0
        
        logger.info("Optuna tuning complete. Best AUC: %.4f", study.best_value)
        logger.info("Best params: %s", best_params)

        model = XGBClassifier(**best_params)

        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            logger.error(
                "[%s] XGBoost training failed. error=%s",
                ErrorCode.ML_INSUFFICIENT_DATA, str(exc),
            )
            raise ModelNotFoundError(
                error_code=ErrorCode.ML_INSUFFICIENT_DATA,
                message=f"XGBoost training failed: {exc}",
            ) from exc

        logger.info("Final XGBoost model trained on best hyperparameters.")

        # Step 5: Find best threshold on training set
        y_prob = model.predict_proba(X_test)[:, 1]   # probability of being denied
        best_threshold, best_f1 = find_best_threshold(
            y_train.values, model.predict_proba(X_train)[:, 1]
        )

        # Step 6: Evaluate on test set using the chosen threshold
        y_pred = (y_prob >= best_threshold).astype(int)
        auc    = roc_auc_score(y_test, y_prob)

        logger.info("=== Test Set Evaluation ===")
        logger.info("AUC-ROC=%.4f  Threshold=%.2f", auc, best_threshold)

        # Log classification report line by line (each line is a valid log entry)
        report = classification_report(y_test, y_pred, target_names=["Approved", "Denied"])
        for line in report.splitlines():  # type: ignore[union-attr]  # output_dict=False → always str
            logger.info(line)

        # Step 7: Save model artifact
        model_path = MODELS_DIR / "denial_model.pkl"
        try:
            joblib.dump(model, model_path)
        except Exception as exc:
            logger.error(
                "[%s] Failed to save model. path=%s error=%s",
                ErrorCode.ML_MODEL_NOT_FOUND, model_path, str(exc),
            )
            raise ModelNotFoundError(
                error_code=ErrorCode.ML_MODEL_NOT_FOUND,
                message=f"Failed to save model to {model_path}: {exc}",
            ) from exc
        logger.info("Model saved. path=%s", model_path)

        # Save threshold — predictor.py reads this at startup
        threshold_path = MODELS_DIR / "threshold.json"
        threshold_data = {
            "threshold":  best_threshold,
            "chosen_by":  "f1_maximisation_on_train_set",
            "best_f1":    round(best_f1, 4),
            "auc_roc":    round(auc, 4),
        }
        try:
            with open(threshold_path, "w", encoding="utf-8") as fp:
                json.dump(threshold_data, fp, indent=2)
        except Exception as exc:
            logger.error(
                "[%s] Failed to save threshold. path=%s error=%s",
                ErrorCode.ML_MODEL_NOT_FOUND, threshold_path, str(exc),
            )
            raise ModelNotFoundError(
                error_code=ErrorCode.ML_MODEL_NOT_FOUND,
                message=f"Failed to write threshold.json: {exc}",
            ) from exc
        logger.info("Threshold saved. path=%s value=%.2f", threshold_path, best_threshold)

        # Save feature importance — used by explainer.py for human-readable output
        importance_dict = dict(zip(NUMERIC_FEATURES, model.feature_importances_.tolist()))
        # Sort descending so the most important feature is first
        importance_sorted = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        importance_path = MODELS_DIR / "feature_importance.json"
        try:
            with open(importance_path, "w", encoding="utf-8") as fp:
                json.dump(importance_sorted, fp, indent=2)
        except Exception as exc:
            logger.error(
                "[%s] Failed to save feature importance. path=%s error=%s",
                ErrorCode.ML_MODEL_NOT_FOUND, importance_path, str(exc),
            )
            raise ModelNotFoundError(
                error_code=ErrorCode.ML_MODEL_NOT_FOUND,
                message=f"Failed to write feature_importance.json: {exc}",
            ) from exc
        logger.info("Feature importance saved. path=%s", importance_path)
        logger.info("Top feature: %s", next(iter(importance_sorted)))

        logger.info("=== ML Training Pipeline Complete ===")

        return {
            "auc":        round(auc,        4),
            "threshold":  best_threshold,
            "f1":         round(best_f1,    4),
            "model_path": str(model_path),
        }

    except (GoldPipelineError, ModelNotFoundError):
        # Already logged — propagate
        raise
    except Exception as exc:
        logger.error(
            "[%s] Unexpected error in ML training pipeline. error=%s",
            ErrorCode.ML_INSUFFICIENT_DATA, str(exc),
        )
        raise ModelNotFoundError(
            error_code=ErrorCode.ML_INSUFFICIENT_DATA,
            message=f"Unexpected training error: {exc}",
        ) from exc


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # APP_ENV guard: suppress console output in production
    is_production = os.getenv("APP_ENV", "development") == "production"

    try:
        metrics = train_model()
    except (GoldPipelineError, ModelNotFoundError) as exc:
        logger.error("Training failed. code=%s error=%s", exc.error_code, exc.message)
        raise SystemExit(1) from exc

    if not is_production:
        print("\nTraining Complete:")
        print(f"  AUC-ROC   : {metrics['auc']}")
        print(f"  Best F1   : {metrics['f1']}")
        print(f"  Threshold : {metrics['threshold']}")
        print(f"  Model     : {metrics['model_path']}")
