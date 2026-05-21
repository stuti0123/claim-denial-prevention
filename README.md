# 🏥 AI-Powered Claim Denial Prevention & Remediation System

A production-grade, end-to-end healthcare claim denial prediction system built on a **local-first, HIPAA-compliant** architecture.

> **Stack:** Python 3.11 · FastAPI · Streamlit · XGBoost (Optuna-tuned) · SHAP · FAISS · sentence-transformers · Pandas Medallion Pipeline

---

## 🏗️ Architecture

```
Raw Data → Bronze Layer → Silver Layer → Gold Layer → ML Model (XGBoost + SHAP)
                                                            ↓
                                                    RAG Engine (FAISS + Policies)
                                                            ↓
                                                  Remediation Agent (Report)
                                                            ↓
                                              FastAPI Backend ← Streamlit Dashboard
```

| Layer | Description |
|---|---|
| **Bronze** | Raw ingestion, schema validation, timestamping |
| **Silver** | Data cleaning, compliance flag injection |
| **Gold** | Feature engineering (7 ML-ready features) |
| **ML Model** | XGBoost + Optuna hyperparameter tuning, AUC-ROC: 0.9374 |
| **RAG** | FAISS vector store + local sentence-transformers (100% offline) |
| **Agent** | Generates structured remediation plan with policy evidence |
| **API** | FastAPI with OAuth2 security layer |
| **Dashboard** | Streamlit with Hybrid AI engine toggle |

---

## ⚡ Quick Start (On Any Device)

### Prerequisites
- Python 3.11+
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/claim-denial-system.git
cd claim-denial-system
```

### Step 2: Create & Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ The first install downloads ~1.5GB (sentence-transformers model). Be patient.

### Step 4: Download the Embedding Model (One Time)
```bash
python scripts/download_model.py
```

### Step 5: Build the RAG Vector Store (One Time)
```bash
python -m src.rag.vector_store
```

### Step 6: Run the Pipeline (Optional — data already in repo)
> **Skip this step if you just want to run the dashboard.**
> Pre-built Bronze/Silver/Gold CSVs and trained model files are included in the repo.

```bash
# Only needed if you want to re-train from scratch:
python -m src.ingestion.bronze_loader
python -m src.silver.validator
python -m src.gold.feature_engineer
python -m src.ml.trainer
```

### Step 7: Start the Backend API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
The API will be live at: http://localhost:8000
Interactive docs at: http://localhost:8000/docs

### Step 8: Start the Dashboard (in a new terminal)
```bash
source venv/bin/activate   # activate venv again in the new terminal
streamlit run src/dashboard/app.py
```
The dashboard will open at: http://localhost:8501

---

## 🔐 OAuth2 Security Demo

The API has a built-in OAuth2 token endpoint. You can test it at:

- **POST** `http://localhost:8000/api/v1/token`
  - `username`: `admin`
  - `password`: `password123`
- **GET** `http://localhost:8000/api/v1/secure-demo` ← Protected endpoint

---

## 🧪 Running Tests
```bash
source venv/bin/activate
python -m pytest src/tests/ -v
```

---

## 📁 Project Structure
```
claim-denial-system/
├── data/
│   ├── bronze/          # Bronze-layer CSVs
│   ├── silver/          # Silver-layer CSVs
│   ├── gold/            # Gold feature table
│   └── policies/        # Policy text files for RAG
├── models/
│   ├── denial_model.pkl        # Trained XGBoost model
│   ├── threshold.json          # Optimal decision threshold
│   ├── feature_medians.json    # Imputation medians
│   ├── policy_index.faiss      # FAISS vector index
│   └── policy_chunks.json      # Policy text chunks
├── src/
│   ├── api/             # FastAPI backend (main.py)
│   ├── dashboard/       # Streamlit frontend (app.py)
│   ├── ingestion/       # Bronze layer loader
│   ├── silver/          # Silver layer validator
│   ├── gold/            # Gold feature engineer
│   ├── ml/              # ML trainer, predictor, explainer
│   ├── rag/             # FAISS vector store + retriever
│   ├── agent/           # Remediation agent + prompts
│   ├── profiling/       # Data profiling module
│   └── core/            # Logger, error codes, exceptions
├── scripts/             # Setup utilities
├── requirements.txt
└── README.md
```

---

## 🩺 Key Features
- ✅ **Medallion Architecture** (Bronze → Silver → Gold)
- ✅ **Hybrid AI Engine** — XGBoost ML + deterministic compliance overrides
- ✅ **SHAP Explainability** — per-claim denial reason breakdown
- ✅ **100% Offline RAG** — FAISS + local sentence-transformers (HIPAA safe)
- ✅ **OAuth2 Security** — FastAPI token endpoint ready
- ✅ **6 Unit Test Suites** — full layer coverage
- ✅ **Optuna Hyperparameter Tuning** — AUC-ROC: 0.9374
