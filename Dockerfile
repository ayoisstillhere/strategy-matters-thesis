# Strategy Matters — Interactive Demo
# Multi-stage build: Node (frontend) + Python (backend)

# ── Stage 1: Build frontend ──
FROM node:20-slim AS frontend-build
WORKDIR /app/demo/frontend
COPY demo/frontend/package.json demo/frontend/package-lock.json* ./
RUN npm ci --ignore-scripts
COPY demo/frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + serve frontend ──
FROM python:3.11-slim
WORKDIR /app

# Install CPU-only PyTorch first (smaller than default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY src/ ./src/
COPY demo/backend/ ./demo/backend/
COPY demo/__init__.py ./demo/__init__.py
COPY data/embeddings/ ./data/embeddings/

# Create runs/demo/ directory for session persistence
RUN mkdir -p runs/demo

# Copy built frontend from stage 1
COPY --from=frontend-build /app/demo/frontend/dist ./demo/frontend/dist

# Expose port (Railway sets PORT env var)
EXPOSE 8000

# Start server — Railway provides $PORT
CMD ["sh", "-c", "uvicorn demo.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
