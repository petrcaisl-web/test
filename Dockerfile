# ---------------------------------------------------------------------------
# Stage 1: Builder — install dependencies into a virtual environment
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tooling
RUN pip install --no-cache-dir hatchling

# Copy only the files needed to resolve dependencies
COPY pyproject.toml ./
COPY app/ ./app/

# Create a virtual environment and install the package with all runtime deps
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: Runtime — minimal image with only what is needed to run the app
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY app/ ./app/

# Ensure the virtual-env binaries take precedence
ENV PATH="/opt/venv/bin:$PATH"

# Cloud Run convention
ENV PORT=8080

# Switch to non-root user
USER appuser

# Health check using the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Start the application
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
