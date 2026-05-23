"""
src/dashboard/app.py
--------------------
Streamlit Dashboard for ClaimOps AI Engine.

Features:
- Login screen with JWT auth (admin / billing_clerk roles)
- Tab 1: Claim Evaluation Simulator (manual form + CSV upload)
- Tab 2: Model Health & Performance
- Tab 3: Claims History (role-aware)
"""

import io

import pandas as pd
import requests
import streamlit as st
import json

# Must be first Streamlit command
st.set_page_config(
    page_title="ClaimOps AI Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
#  Config                                                                       #
# --------------------------------------------------------------------------- #

import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --------------------------------------------------------------------------- #
#  CSS                                                                          #
# --------------------------------------------------------------------------- #

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0;
    }
    .sub-header { color: #64748B; font-size: 1.05rem; margin-top: 0; margin-bottom: 24px; }
    [data-testid="stSidebar"] { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
    .metric-label { font-size: 0.78rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #0F172A; }
    .login-container { max-width: 420px; margin: 80px auto; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Auth Helpers                                                                 #
# --------------------------------------------------------------------------- #

def do_login(username: str, password: str) -> bool:
    """POST to /api/v1/token, store token + role in session state."""
    try:
        resp = requests.post(
            f"{API_URL}/api/v1/token",
            data={"username": username, "password": password},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state["token"] = data["access_token"]
            st.session_state["role"] = data["role"]
            st.session_state["username"] = data["username"]
            return True
        return False
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the FastAPI backend. Please ensure it is running.")
        return False


def auth_headers() -> dict:
    """Return Bearer token header for API calls."""
    return {"Authorization": f"Bearer {st.session_state.get('token', '')}"}


# --------------------------------------------------------------------------- #
#  LOGIN SCREEN                                                                 #
# --------------------------------------------------------------------------- #

if "token" not in st.session_state:
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("## 🏥 ClaimOps AI Engine")
    st.markdown("**Sign in to continue**")
    st.divider()

    st.markdown("Please enter your authorized corporate credentials to access the HIPAA-compliant dashboard.")
    st.divider()

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
        if submitted:
            if do_login(username, password):
                st.rerun()
            else:
                st.error("Invalid credentials. Please check your username and password.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()  # Block everything below until logged in


# --------------------------------------------------------------------------- #
#  AUTHENTICATED: Main App                                                      #
# --------------------------------------------------------------------------- #

st.markdown(f'<p class="main-header">ClaimOps AI Engine</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-header">Automated Claims Prediction & RAG Remediation — '
    f'Logged in as <strong>{st.session_state["username"]}</strong> '
    f'({st.session_state["role"].replace("_", " ").title()})</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
#  SIDEBAR: System Telemetry & Logout                                           #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("System Telemetry")
    st.divider()

    try:
        resp = requests.get(f"{API_URL}/api/v1/analytics/summary", headers=auth_headers(), timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.markdown(f'<div class="metric-label">Total Processed Claims</div>'
                        f'<div class="metric-value">{data["total_claims"]:,}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Total Billed Volume</div>'
                        f'<div class="metric-value">₹{data["total_billed_amount"]:,.0f}</div>', unsafe_allow_html=True)
        else:
            st.warning("Analytics unavailable.")
    except Exception:
        st.error("FastAPI Backend Offline")

    st.divider()

    try:
        with open("models/threshold.json") as f:
            metrics = json.load(f)
        st.markdown(f'<div class="metric-label">Active Model</div>'
                    f'<div class="metric-value">XGBoost-Optuna</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">AUC-ROC</div>'
                    f'<div class="metric-value">{metrics.get("auc_roc", 0.9374):.4f}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Decision Threshold</div>'
                    f'<div class="metric-value">{metrics.get("threshold", 0.37):.2f}</div>', unsafe_allow_html=True)
    except Exception:
        st.warning("Model metrics unavailable.")

    st.divider()
    if st.button("Sign Out", use_container_width=True):
        for key in ["token", "role", "username"]:
            st.session_state.pop(key, None)
        st.rerun()


# --------------------------------------------------------------------------- #
#  TABS                                                                         #
# --------------------------------------------------------------------------- #

tab1, tab2, tab3 = st.tabs([
    "Claim Evaluation Simulator",
    "Model Health & Performance",
    "Claims History",
])


# ─────────────────────────────────────────────────────────────────────────── #
#  TAB 1: Claim Simulator                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

with tab1:
    st.markdown("Enter claim details manually or **upload a CSV file** to pre-fill the form.")

    # --- CSV Upload ---
    uploaded_file = st.file_uploader(
        "Upload Claim CSV (optional — auto-fills form fields)",
        type=["csv"],
        help="CSV must have columns: claim_id, patient_id, provider_id, diagnosis_code, procedure_code, billed_amount",
    )

    # Pre-fill defaults (overridden by CSV if uploaded)
    defaults = {
        "claim_id": "CLM-TEST-001",
        "patient_id": "PAT-555",
        "provider_id": "PRV-101",
        "billed_amount": 25000.0,
        "diag_code": "",
        "proc_code": "",
    }

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(io.StringIO(uploaded_file.read().decode("utf-8")))
            if len(df_upload) > 0:
                row = df_upload.iloc[0]
                defaults["claim_id"] = str(row.get("claim_id", defaults["claim_id"]))
                defaults["patient_id"] = str(row.get("patient_id", defaults["patient_id"]))
                defaults["provider_id"] = str(row.get("provider_id", defaults["provider_id"]))
                defaults["billed_amount"] = float(row.get("billed_amount", defaults["billed_amount"]) or 0)
                defaults["diag_code"] = str(row.get("diagnosis_code", "") or "")
                defaults["proc_code"] = str(row.get("procedure_code", "") or "")
                st.success(f"CSV loaded: {len(df_upload)} claim(s) found. Showing first row.")
                if len(df_upload) > 1:
                    st.info(f"Note: Only the first row is loaded into the simulator. Full batch processing is a future feature.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    # --- Form ---
    with st.container(border=True):
        with st.form("predict_form", border=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                claim_id = st.text_input("Claim ID", value=defaults["claim_id"])
                patient_id = st.text_input("Patient ID", value=defaults["patient_id"])
            with col2:
                provider_id = st.text_input("Provider ID", value=defaults["provider_id"])
                billed_amount = st.number_input("Billed Amount (₹)", value=defaults["billed_amount"], step=1000.0)
            with col3:
                diag_code = st.text_input("Diagnosis Code (ICD-10)", value=defaults["diag_code"],
                                           placeholder="e.g. J18.9 — leave blank to simulate missing")
                proc_code = st.text_input("Procedure Code (CPT)", value=defaults["proc_code"],
                                           placeholder="e.g. 99213 — leave blank to simulate missing")

            st.markdown("<br>", unsafe_allow_html=True)
            strict_mode = st.toggle(
                "Enable Hybrid AI (Strict Compliance Overrides)",
                value=True,
                help="ON = business rules override ML for obvious violations. OFF = pure ML probability.",
            )
            submitted = st.form_submit_button("Run AI Analysis", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "claim_id": claim_id,
            "patient_id": patient_id,
            "provider_id": provider_id,
            "billed_amount": billed_amount,
            "diagnosis_code": diag_code if diag_code.strip() else None,
            "procedure_code": proc_code.strip() if proc_code.strip() else None,
            "claim_date": "2024-01-01",
            "strict_mode": strict_mode,
        }

        with st.spinner("Analysing claim — XGBoost inference + FAISS policy search in progress..."):
            try:
                resp = requests.post(
                    f"{API_URL}/api/v1/predict",
                    json=payload,
                    headers=auth_headers(),
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    pred = data["prediction"]
                    plan = data["remediation_plan"]
                    st.divider()

                    if pred["is_denied"]:
                        colA, colB = st.columns([1, 1.5])
                        with colA:
                            st.error(f"⚠️ **HIGH DENIAL RISK**\n\nProbability: {pred['denial_probability'] * 100:.1f}%")
                            st.markdown("##### Top Denial Reasons (Simulated SHAP Log-Odds)")
                            for c in pred.get("feature_contributions", []):
                                with st.expander(f"🔴 {c['feature']} — SHAP score {c['score']} (↑ increases risk)"):
                                    st.write(f"**How to fix:** {c['suggestion']}")
                        with colB:
                            st.info("📄 **Remediation Plan Generated**")
                            st.markdown("##### Policy Evidence (RAG Retrieval)")
                            if plan.get("rag_evidence"):
                                st.dataframe(pd.DataFrame(plan["rag_evidence"]), hide_index=True, use_container_width=True)
                            st.markdown("##### Full Remediation Report")
                            st.markdown(plan["full_report"], unsafe_allow_html=True)
                    else:
                        st.success(f"✅ **CLEAN CLAIM**  —  Denial Probability: {pred['denial_probability'] * 100:.1f}%")
                        st.write("No compliance or billing violations detected. This claim is ready for submission.")

                elif resp.status_code == 401:
                    st.error("Session expired. Please sign out and log in again.")
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Please ensure the backend is running.")


# ─────────────────────────────────────────────────────────────────────────── #
#  TAB 2: Model Health                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

with tab2:
    st.subheader("ML Model — Performance, Explainability & Predictions")
    try:
        with open("models/threshold.json") as f:
            metrics_data = json.load(f)

        m1, m2, m3 = st.columns(3)
        m1.metric("Model", "XGBoost Classifier")
        m2.metric("ROC-AUC Score", f"{metrics_data.get('auc_roc', 0.9374):.4f}")
        m3.metric("F1 Score (Train)", f"{metrics_data.get('best_f1', 0.8633):.4f}")

        st.divider()
        st.subheader("Global Feature Importances")

        with open("models/feature_importance.json") as f:
            importance_data = json.load(f)

        df_importance = pd.DataFrame({
            "Feature": list(importance_data.keys()),
            "Importance": list(importance_data.values()),
        }).sort_values("Importance", ascending=True)

        st.bar_chart(df_importance.set_index("Feature"), horizontal=True)

    except Exception as e:
        st.warning(f"Could not load model metrics: {e}")


# ─────────────────────────────────────────────────────────────────────────── #
#  TAB 3: Claims History                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

with tab3:
    role = st.session_state.get("role", "billing_clerk")
    if role == "billing_admin":
        st.subheader("All Claims — Admin View")
        st.caption("You are viewing predictions submitted by all users.")
    else:
        st.subheader("My Claims History")
        st.caption("Showing your last 100 submitted claim evaluations.")

    if st.button("Refresh History"):
        st.rerun()

    try:
        resp = requests.get(
            f"{API_URL}/api/v1/claims/history",
            headers=auth_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            history = resp.json().get("history", [])
            if history:
                df_hist = pd.DataFrame(history)

                # Format columns for readability
                df_hist["denial_probability"] = df_hist["denial_probability"].map(lambda x: f"{x * 100:.1f}%" if x is not None else "—")
                df_hist["is_denied"] = df_hist["is_denied"].map(lambda x: "🔴 Denied" if x else "🟢 Clean")
                df_hist["submitted_at"] = pd.to_datetime(df_hist["submitted_at"]).apply(lambda x: x.strftime("%Y-%m-%d %H:%M") if pd.notnull(x) else "")

                display_cols = ["claim_id", "patient_id", "provider_id", "billed_amount",
                                "diagnosis_code", "procedure_code", "risk_level",
                                "denial_probability", "is_denied", "flags", "submitted_at"]
                if role == "billing_admin":
                    display_cols.insert(-1, "submitted_by")

                st.dataframe(
                    df_hist[[c for c in display_cols if c in df_hist.columns]],
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(f"Showing {len(df_hist)} records.")
            else:
                st.info("No claim evaluations submitted yet. Go to the Claim Simulator tab to submit one.")

        elif resp.status_code == 401:
            st.error("Session expired. Please sign out and log in again.")
        else:
            st.error(f"Could not load history: {resp.text}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Please ensure the backend is running.")
