# 🏥 AI-Powered Claim Denial Prevention & Remediation System

A production-grade, end-to-end healthcare claim denial prediction system built on a **local-first, HIPAA-compliant** architecture.

> **Stack:** Python 3.11 · FastAPI · Streamlit · XGBoost (Optuna-tuned) · SHAP · FAISS · PostgreSQL · Docker · Nginx

---

## 🩺 Key Features
- ✅ **Medallion Architecture** (Bronze → Silver → Gold) for robust data pipelines.
- ✅ **Hybrid AI Engine** — XGBoost ML + deterministic compliance overrides.
- ✅ **SHAP Explainability** — Per-claim denial reason breakdown.
- ✅ **100% Offline RAG** — FAISS + local sentence-transformers (HIPAA safe, no data leaves the server).
- ✅ **Role-Based Access Control (RBAC)** — Strict separation between Billing Admins and Clerks.
- ✅ **Enterprise Cloud Deployment** — Fully containerized with Docker, Nginx Reverse Proxy, and Let's Encrypt HTTPS certificates.
- ✅ **Persistent Relational DB** — Powered by AWS RDS / PostgreSQL.
- ✅ **Optuna Hyperparameter Tuning** — ML AUC-ROC: 0.9374.

---

## 🏗️ Architecture

```
Raw Data → Bronze Layer → Silver Layer → Gold Layer → ML Model (XGBoost + SHAP)
                                                            ↓
                                                    RAG Engine (FAISS + Policies)
                                                            ↓
                                                  Remediation Agent (Report)
                                                            ↓
                          PostgreSQL DB ← FastAPI Backend ← Streamlit Dashboard
```

| Layer | Description |
|---|---|
| **Bronze** | Raw ingestion, schema validation, timestamping |
| **Silver** | Data cleaning, compliance flag injection |
| **Gold** | Feature engineering (7 ML-ready features) |
| **ML Model** | XGBoost + Optuna hyperparameter tuning |
| **RAG** | FAISS vector store + local sentence-transformers (100% offline) |
| **Agent** | Generates structured remediation plan with policy evidence |
| **API** | FastAPI with OAuth2 RBAC security layer |
| **Dashboard** | Streamlit with Hybrid AI engine toggle |
| **Infrastructure**| Docker Compose, Nginx Reverse Proxy, Certbot HTTPS |

---

## ☁️ Cloud Deployment (AWS EC2)

The system is designed for secure, HIPAA-compliant enterprise deployment using **Docker Compose**.

### Running the Production Stack:
1. Clone the repository on your server.
2. Ensure you have Docker and Docker Compose installed.
3. Configure your `.env` file with your PostgreSQL `DATABASE_URL` and security keys.
4. Launch the stack:
   ```bash
   docker-compose up -d --build
   ```
5. Configure Nginx and Certbot to serve the application securely over HTTPS (port 443).

---

## ⚡ Local Quick Start (Development)

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

### Step 4: Download the Embedding Model & Build FAISS Index (One Time)
```bash
python scripts/download_model.py
python -m src.rag.vector_store
```

### Step 5: Start the Backend API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Interactive docs at: http://localhost:8000/docs

### Step 6: Start the Dashboard (in a new terminal)
```bash
source venv/bin/activate
streamlit run src/dashboard/app.py
```
The dashboard will open at: http://localhost:8501

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
│   ├── policy_index.faiss      # FAISS vector index
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
│   └── core/            # Logger, error codes, database models
├── scripts/             # Setup utilities
├── docker-compose.yml   # Production container orchestration
├── Dockerfile           # API container build
├── Dockerfile.streamlit # Dashboard container build
├── requirements.txt     # Python dependencies
└── README.md
```
