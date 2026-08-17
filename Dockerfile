# Dockerfile — FastAPI Claims Scoring Service
# ==============================================
# Multi-stage build for the claims audit scoring API.
# Uses Python 3.12-slim for a small production image.
#
# Build:  docker-compose build api
# Run:    docker-compose up -d api

FROM python:3.12-slim AS base

# System deps for cassandra-driver C extensions and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libev-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run uvicorn — 1 worker for single-process model serving;
# increase workers if using gunicorn + uvicorn.workers
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
