# ============================================================
# Dockerfile — FastAPI Backend
# Build: docker build -t claimops-api .
# Run:   docker run -p 8000:8000 --env-file .env claimops-api
# ============================================================

FROM python:3.11-slim

# Suppress pip warnings
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for FAISS + XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Expose API port
EXPOSE 8000

# Run FastAPI via Gunicorn with Uvicorn workers (production-grade concurrency)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
