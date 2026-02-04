# ==============================================================================
# POW Agent - Production Dockerfile
# Multi-stage build for optimal image size and security
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and build
# ------------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# ------------------------------------------------------------------------------
FROM python:3.11-slim as production

# Security: Run as non-root user
RUN groupadd -r powagent && useradd -r -g powagent powagent

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy application code
COPY --chown=powagent:powagent agent/ ./agent/
COPY --chown=powagent:powagent api/ ./api/
COPY --chown=powagent:powagent colosseum/ ./colosseum/
COPY --chown=powagent:powagent solana/ ./solana/
COPY --chown=powagent:powagent tasks/ ./tasks/
COPY --chown=powagent:powagent prompts/ ./prompts/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R powagent:powagent /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=4 \
    LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health/live || exit 1

# Expose port
EXPOSE ${PORT}

# Switch to non-root user
USER powagent

# Entrypoint - Run the API server
CMD ["sh", "-c", "uvicorn api.server:app --host ${HOST} --port ${PORT} --workers ${WORKERS} --log-level ${LOG_LEVEL}"]

# ------------------------------------------------------------------------------
# Stage 3: Development - With dev tools
# ------------------------------------------------------------------------------
FROM production as development

USER root

# Install dev dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    ruff \
    mypy

USER powagent

# Override command for development
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
