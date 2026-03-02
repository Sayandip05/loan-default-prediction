# Multi-stage Dockerfile for Loan Default Prediction

# Base image with Python
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# ----------------- API Stage -----------------
FROM base as api

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ----------------- Frontend Stage -----------------
FROM base as frontend

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ----------------- MLflow Stage -----------------
FROM base as mlflow

# Expose MLflow port
EXPOSE 5000

# Create MLflow directory
RUN mkdir -p /app/mlruns /app/mlartifacts

# Run MLflow server
CMD ["mlflow", "server", \
    "--backend-store-uri", "sqlite:///mlflow.db", \
    "--default-artifact-root", "/app/mlartifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000"]
