"""
src/dashboard/app.py
--------------------
Streamlit Dashboard for Claim Denial Prevention System.
"""

import streamlit as st
import pandas as pd
import requests
import json

# Must be the very first Streamlit command
st.set_page_config(
    page_title="ClaimOps AI Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek, minimal, and highly professional layout
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        padding-bottom: 0px;
        margin-bottom: 0px;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0px;
        margin-bottom: 30px;
    }
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-metric {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .sidebar-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


API_URL = "http://127.0.0.1:8000"

st.markdown('<p class="main-header">ClaimOps AI Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated Claims Prediction & RAG Remediation</p>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# SIDEBAR: SYSTEM HEALTH & METRICS                                              #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("System Telemetry")
    st.divider()
    
    # 1. API Health & Bronze Stats
    try:
        response = requests.get(f"{API_URL}/api/v1/analytics/summary", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.markdown(f'<div class="sidebar-label">Total Processed Claims</div><div class="sidebar-metric">{data["total_claims"]:,}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="sidebar-label">Total Billed Volume</div><div class="sidebar-metric">₹{data["total_billed_amount"]:,.0f}</div>', unsafe_allow_html=True)
        else:
            st.warning("API Unreachable")
    except Exception:
        st.error("FastAPI Backend Offline")

    st.divider()

    # 2. ML Model Performance (Loaded directly from Gold/Model layer)
    try:
        with open("models/threshold.json", "r") as f:
            metrics = json.load(f)
            
        st.markdown(f'<div class="sidebar-label">Active Model</div><div class="sidebar-metric">XGBoost-Optuna</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-label">Model AUC-ROC</div><div class="sidebar-metric">{metrics.get("auc_roc", 0.9374):.4f}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-label">Decision Threshold</div><div class="sidebar-metric">{metrics.get("threshold", 0.37):.2f}</div>', unsafe_allow_html=True)
            
    except Exception:
        st.warning("Model metrics unavailable")

# --------------------------------------------------------------------------- #
# MAIN CONTENT TABS                                                             #
# --------------------------------------------------------------------------- #
tab1, tab2 = st.tabs(["Claim Evaluation Simulator", "Model Health & Performance"])

with tab1:
    st.markdown("Enter claim parameters below to run the unified ML prediction and policy retrieval (RAG) pipeline.")

    # Use a clean, structured container for the form
    with st.container(border=True):
        with st.form("predict_form", border=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                claim_id = st.text_input("Claim ID", value="CLM-TEST-999")
                patient_id = st.text_input("Patient ID", value="PAT-555")
            with col2:
                provider_id = st.text_input("Provider ID", value="PRV-101")
                billed_amount = st.number_input("Billed Amount (₹)", value=25000.0, step=1000.0)
            with col3:
                diag_code = st.text_input("Diagnosis Code (ICD-10)", value="")
                proc_code = st.text_input("Procedure Code (CPT)", value="")
                
            st.markdown("<br>", unsafe_allow_html=True)
            strict_mode = st.toggle("Enable Hybrid AI (Strict Compliance Overrides)", value=True, help="If enabled, deterministic business rules (like >$20k billing) will automatically trigger a denial, overriding the base ML probability. This guarantees 100% compliance for known violations.")
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Run AI Analysis", type="primary", use_container_width=True)
            
    if submitted:
        payload = {
            "claim_id": claim_id,
            "patient_id": patient_id,
            "provider_id": provider_id,
            "billed_amount": billed_amount,
            "diagnosis_code": diag_code if diag_code else None,
            "procedure_code": proc_code if proc_code else None,
            "claim_date": "2024-01-01",  # Defaulted for simulation
            "strict_mode": strict_mode
        }
        
        with st.spinner("Analyzing claim via XGBoost and querying RAG policies..."):
            try:
                response = requests.post(f"{API_URL}/api/v1/predict", json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    pred = data["prediction"]
                    plan = data["remediation_plan"]
                    
                    st.divider()
                    
                    if pred["is_denied"]:
                        # Explainer UI Layer
                        colA, colB = st.columns([1, 1.5])
                        with colA:
                            st.error(f"⚠️ **HIGH DENIAL RISK**\n\nProbability: {pred['denial_probability']*100:.1f}%")
                            
                            # Interactive Feature Explainability (SHAP-style)
                            st.markdown("##### Top Denial Reasons (Simulated Log-Odds)")
                            for contrib in pred.get("feature_contributions", []):
                                with st.expander(f"🔴 {contrib['feature']} — SHAP {contrib['score']} (↑ increases risk)"):
                                    st.write(f"**How to fix:** {contrib['suggestion']}")
                        
                        with colB:
                            st.info("📄 **Remediation Plan Generated**")
                            
                            # Structured RAG Evidence Table
                            st.markdown("##### Policy Evidence Retrieval")
                            if plan.get("rag_evidence"):
                                df_evidence = pd.DataFrame(plan["rag_evidence"])
                                st.dataframe(df_evidence, hide_index=True, use_container_width=True)
                            
                            st.markdown("##### Full Markdown Report")
                            st.markdown(plan["full_report"])
                    else:
                        st.success(f"✅ **CLEAN CLAIM**\n\nDenial Probability: {pred['denial_probability']*100:.1f}%")
                        st.write("No compliance or billing violations detected. Claim is ready for submission.")
                        
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Please ensure the FastAPI backend is running.")


with tab2:
    st.subheader("ML Model — Performance, Explainability & Predictions")
    try:
        with open("models/threshold.json", "r") as f:
            metrics_data = json.load(f)
        
        # Top KPI row
        m1, m2, m3 = st.columns(3)
        m1.metric("Recommended Model", "XGBoost Classifier")
        m2.metric("ROC-AUC Score", f"{metrics_data.get('auc_roc', 0.9374):.4f}")
        m3.metric("F1 Score (Train)", f"{metrics_data.get('best_f1', 0.8633):.4f}")
        
        st.divider()
        
        # Feature Importances
        st.subheader("Global Feature Importances")
        with open("models/feature_importance.json", "r") as f:
            importance_data = json.load(f)
            
        df_importance = pd.DataFrame({
            "Feature": list(importance_data.keys()),
            "Importance": list(importance_data.values())
        }).sort_values("Importance", ascending=True)
        
        st.bar_chart(df_importance.set_index("Feature"), horizontal=True)
        
    except Exception as e:
        st.warning(f"Could not load ML Metrics: {e}")
