# ─────────────────────────────────────────────────────────────────────────────
# FlowGuard AI – Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t flowguard-ai .
# Run:    docker run -p 8000:8000 flowguard-ai
# ─────────────────────────────────────────────────────────────────────────────

# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only the lean prod requirements for Docker layer caching
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app/ ./app/
COPY models/ ./models/

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
ENV PORT=8000

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
