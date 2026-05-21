"""
Week 2 — Streamlit Dashboard
----------------------------
Displays basic analytics by fetching data from the backend API.
"""

import streamlit as st
import requests
import pandas as pd
import altair as alt

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Week 2 Analytics Dashboard", layout="wide")

st.title("🏥 Claim Denial System - Analytics Dashboard")
st.markdown("This dashboard fetches data dynamically from the independent FastAPI backend.")

@st.cache_data(ttl=60)
def fetch_summary():
    try:
        response = requests.get(f"{API_URL}/api/v1/analytics/summary")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch summary: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_provider_activity():
    try:
        response = requests.get(f"{API_URL}/api/v1/analytics/provider-activity")
        response.raise_for_status()
        return response.json().get("provider_activity", [])
    except Exception as e:
        st.error(f"Failed to fetch provider activity: {e}")
        return []

@st.cache_data(ttl=60)
def fetch_claims_trend():
    try:
        response = requests.get(f"{API_URL}/api/v1/analytics/claims-trend")
        response.raise_for_status()
        return response.json().get("claims_trend", [])
    except Exception as e:
        st.error(f"Failed to fetch claims trend: {e}")
        return []

# Fetch Data
summary = fetch_summary()
provider_activity = fetch_provider_activity()
claims_trend = fetch_claims_trend()

# Layout: Summary KPIs
if summary:
    st.header("KPI Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Claims", f"{summary.get('total_claims', 0):,}")
    col2.metric("Total Billed Amount", f"${summary.get('total_billed_amount', 0):,.2f}")
    col3.metric("Unique Providers", f"{summary.get('unique_providers', 0):,}")
    st.markdown("---")

col_left, col_right = st.columns(2)

# Layout: Past Claims History
with col_left:
    st.subheader("Past Claims History (Trend)")
    if claims_trend:
        df_trend = pd.DataFrame(claims_trend)
        # Rename columns for better tooltips
        df_trend = df_trend.rename(columns={"date_parsed": "Date", "billed_amount": "Total Billed ($)"})
        
        line_chart = alt.Chart(df_trend).mark_line(point=True, color="#1f77b4").encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Total Billed ($):Q", title="Billed Amount ($)"),
            tooltip=["Date", "Total Billed ($)"]
        ).interactive()
        
        st.altair_chart(line_chart, use_container_width=True)
    else:
        st.info("No trend data available.")

# Layout: Provider Activity
with col_right:
    st.subheader("Provider Activity (Top 10)")
    if provider_activity:
        df_activity = pd.DataFrame(provider_activity)
        df_activity = df_activity.rename(columns={"doctor_name": "Provider", "claim_count": "Claim Count"})
        
        bar_chart = alt.Chart(df_activity).mark_bar(color="#ff7f0e").encode(
            x=alt.X("Claim Count:Q", title="Number of Claims"),
            y=alt.Y("Provider:N", sort="-x", title="Provider"),
            tooltip=["Provider", "Claim Count"]
        )
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.info("No provider activity data available.")
