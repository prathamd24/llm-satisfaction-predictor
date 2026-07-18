# ─────────────────────────────────────────────────────────────────────────────
# LLM Satisfaction Predictor — Dockerfile
# Runs the FastAPI backend with Uvicorn.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system-level build dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/  ./src/
COPY app/  ./app/

# Copy pre-trained model and artifacts (must be built with `python src/train.py` first)
COPY models/     ./models/
COPY artifacts/  ./artifacts/

# Create data directories (CSV is not bundled in the image)
RUN mkdir -p data/raw data/processed

# Expose the API port
EXPOSE 8000

# Health check — confirms the /health endpoint is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
