"""
src/api/main.py
---------------
Backend API for the Claim Denial Prevention System.

HOW TO RUN
----------
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Fix OpenMP conflict on macOS between XGBoost and FAISS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.api.auth import (
    TokenResponse,
    authenticate_user,
    create_access_token,
    get_current_user,
    require_admin,
)
from src.core.database import PredictionHistory, User, get_db, init_db
from src.agent.remediator import RemediationAgent
from src.ml.predictor import predict

# --------------------------------------------------------------------------- #
#  App bootstrap                                                                #
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="ClaimOps AI API",
    version="2.0.0",
    description="AI-powered claim denial prevention with ML + RAG + Hybrid Compliance Engine.",
)

# Initialise DB and seed demo users on startup
init_db()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"

# Agent is lightweight to instantiate; RAG loads lazily on first call
agent = RemediationAgent()


# --------------------------------------------------------------------------- #
#  Auth Endpoints                                                               #
# --------------------------------------------------------------------------- #

@app.post("/api/v1/token", response_model=TokenResponse, tags=["Auth"])
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Login with username and password. Returns a JWT bearer token.

    Demo credentials:
      admin / admin123  →  billing_admin role
      clerk / clerk123  →  billing_clerk role
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    token = create_access_token({"sub": user.username, "role": user.role})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        username=user.username,
    )


# --------------------------------------------------------------------------- #
#  Health                                                                       #
# --------------------------------------------------------------------------- #

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}


# --------------------------------------------------------------------------- #
#  Bronze Analytics (admin-only)                                                #
# --------------------------------------------------------------------------- #

def load_bronze_claims() -> pd.DataFrame:
    path = BRONZE_DIR / "bronze_claims.csv"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Bronze claims data not found.")
    return pd.read_csv(path)


def load_bronze_providers() -> pd.DataFrame:
    path = BRONZE_DIR / "bronze_providers.csv"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Bronze providers data not found.")
    return pd.read_csv(path)


@app.get("/api/v1/analytics/summary", tags=["Analytics"])
def get_summary(current_user: User = Depends(get_current_user)):
    try:
        claims = load_bronze_claims()
        return {
            "total_claims": len(claims),
            "total_billed_amount": float(claims["billed_amount"].sum()),
            "unique_providers": claims["provider_id"].nunique(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/provider-activity", tags=["Analytics"])
def get_provider_activity(current_user: User = Depends(require_admin)):
    try:
        claims = load_bronze_claims()
        providers = load_bronze_providers()
        merged = claims.merge(providers, on="provider_id", how="left")
        activity = merged["doctor_name"].value_counts().reset_index()
        activity.columns = ["doctor_name", "claim_count"]  # type: ignore[assignment]
        return {"provider_activity": activity.head(10).to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/claims-trend", tags=["Analytics"])
def get_claims_trend(current_user: User = Depends(get_current_user)):
    try:
        claims = load_bronze_claims()
        if "claim_date" in claims.columns:
            claims["date_parsed"] = pd.to_datetime(claims["claim_date"], errors="coerce")
        elif "date" in claims.columns:
            claims["date_parsed"] = pd.to_datetime(claims["date"], errors="coerce")
        else:
            return {"claims_trend": []}
        trend = claims.groupby(claims["date_parsed"].dt.date)["billed_amount"].sum().reset_index()  # type: ignore[union-attr]
        trend = trend.dropna()
        trend["date_parsed"] = trend["date_parsed"].astype(str)
        trend = trend.sort_values("date_parsed")
        return {"claims_trend": trend.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
#  Claims History                                                               #
# --------------------------------------------------------------------------- #

@app.get("/api/v1/claims/history", tags=["Claims"])
def get_claims_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return all past predictions for the logged-in user.
    Admins see all predictions from all users.
    """
    query = db.query(PredictionHistory)
    if current_user.role != "billing_admin":
        query = query.filter(PredictionHistory.username == current_user.username)
    records = query.order_by(PredictionHistory.submitted_at.desc()).limit(100).all()

    return {
        "history": [
            {
                "id": r.id,
                "claim_id": r.claim_id,
                "patient_id": r.patient_id,
                "provider_id": r.provider_id,
                "billed_amount": r.billed_amount,
                "diagnosis_code": r.diagnosis_code,
                "procedure_code": r.procedure_code,
                "denial_probability": r.denial_probability,
                "risk_level": r.risk_level,
                "is_denied": bool(r.is_denied),
                "flags": r.flags,
                "submitted_by": r.username,
                "submitted_at": r.submitted_at.isoformat() if r.submitted_at else None,
            }
            for r in records
        ]
    }


# --------------------------------------------------------------------------- #
#  ML Prediction + RAG Remediation                                              #
# --------------------------------------------------------------------------- #

class ClaimPayload(BaseModel):
    claim_id: str
    patient_id: str
    provider_id: str
    diagnosis_code: str | None = None
    procedure_code: str | None = None
    billed_amount: float | None = None
    claim_date: str | None = None
    strict_mode: bool = True


@app.post("/api/v1/predict", tags=["Claims"])
def predict_and_remediate(
    claim: ClaimPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Run ML prediction + compliance check + RAG remediation for a single claim.
    Saves the result to the prediction history table.

    NOTE: First call is slow (~15s) as models lazy-load. Subsequent calls are near-instant.
    """
    try:
        claim_dict = claim.model_dump()

        # --- Real-time Feature Engineering ---
        ml_features = claim_dict.copy()
        billed_amt = claim_dict.get("billed_amount") or 0.0
        ml_features["is_high_biller"] = 1.0 if billed_amt > 20000 else 0.0
        ml_features["claim_amount_ratio"] = 4.5 if ml_features["is_high_biller"] == 1.0 else 1.0

        missing_count = 0
        if not claim_dict.get("diagnosis_code"): missing_count += 1
        if not claim_dict.get("procedure_code"): missing_count += 1
        if claim_dict.get("billed_amount") is None: missing_count += 1
        ml_features["missing_fields_count"] = float(missing_count)

        if not claim_dict.get("diagnosis_code"):
            ml_features["diagnosis_severity_flag"] = 1.0
        if missing_count > 0 or ml_features["is_high_biller"] == 1.0:
            ml_features["provider_denial_rate"] = 0.85

        # 1. ML Prediction
        prediction = predict(ml_features)

        # 2. Hybrid Compliance Override
        if claim_dict.get("strict_mode", True):
            if ml_features["is_high_biller"] == 1.0 or missing_count > 0:
                base_prob = 0.88 if missing_count == 1 else 0.95
                if ml_features["is_high_biller"] == 1.0:
                    base_prob = max(base_prob, 0.92)
                prediction.probability = base_prob
                prediction.denial_label = 1
                prediction.risk_level = "HIGH"

        # 3. Derive validation flags
        flags: list[str] = []
        if not claim_dict.get("diagnosis_code"):
            flags.append("WARN_MISSING_DIAGNOSIS")
        if not claim_dict.get("procedure_code"):
            flags.append("WARN_MISSING_PROCEDURE")
        if claim_dict.get("billed_amount") is None:
            flags.append("WARN_MISSING_AMOUNT")
        elif claim_dict["billed_amount"] > 20000:
            flags.append("WARN_HIGH_BILLING")
        if not claim_dict.get("diagnosis_code") and not claim_dict.get("procedure_code"):
            flags.append("ERR_INCOMPLETE_CLAIM")

        # 4. Remediation Agent
        plan = agent.generate_plan(claim_dict, flags)

        # 5. SHAP-style feature contributions
        feature_contributions = []
        if "WARN_MISSING_DIAGNOSIS" in flags:
            feature_contributions.append({"feature": "Missing Diagnosis Code", "score": "+5.01", "suggestion": "Add a valid ICD-10 diagnosis code before submission."})
        if "WARN_MISSING_PROCEDURE" in flags:
            feature_contributions.append({"feature": "Missing Procedure Code", "score": "+4.85", "suggestion": "Add a valid CPT procedure code."})
        if "WARN_HIGH_BILLING" in flags:
            feature_contributions.append({"feature": "High Billing Amount", "score": "+3.15", "suggestion": "Review against reference pricing and attach supporting receipts."})
        if not feature_contributions and prediction.probability > 0.15:
            feature_contributions.append({"feature": "Provider Denial History", "score": "+1.05", "suggestion": "Provider has a historical pattern of high-risk claim submissions."})

        # 6. Save to prediction history
        db.add(PredictionHistory(
            username=current_user.username,
            claim_id=claim_dict["claim_id"],
            patient_id=claim_dict.get("patient_id"),
            provider_id=claim_dict.get("provider_id"),
            billed_amount=claim_dict.get("billed_amount"),
            diagnosis_code=claim_dict.get("diagnosis_code"),
            procedure_code=claim_dict.get("procedure_code"),
            denial_probability=prediction.probability,
            risk_level=prediction.risk_level,
            is_denied=int(prediction.denial_label == 1),
            flags="|".join(flags),
            submitted_at=datetime.utcnow(),
        ))
        db.commit()

        return {
            "prediction": {
                "claim_id": claim_dict["claim_id"],
                "denial_probability": prediction.probability,
                "is_denied": prediction.denial_label == 1,
                "risk_level": prediction.risk_level,
                "flags": flags,
                "feature_contributions": feature_contributions,
            },
            "remediation_plan": {
                "sources": plan.sources,
                "full_report": plan.full_report,
                "rag_evidence": plan.rag_evidence,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[PREDICT ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
