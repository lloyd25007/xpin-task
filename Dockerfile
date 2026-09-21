# =============================================================================
# One image, two entrypoints: the compose file runs it as the FastAPI service
# and again as the Streamlit UI, overriding `command` each time. Sharing an
# image keeps the dependency set identical across both, which matters because
# the UI imports the same schemas the API serves.
# =============================================================================

FROM python:3.12-slim AS base

# Python behaviour inside containers:
#   UNBUFFERED    - logs appear immediately instead of on flush
#   DONTWRITEBYTECODE - no .pyc clutter in the layer
#   HF_HOME       - keep model downloads on a mounted volume, not in the layer
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/appuser/.cache/huggingface

# System packages:
#   libgl1 / libglib2.0-0 - OpenCV, pulled in by Docling's layout model
#   poppler-utils         - PDF rasterisation used during parsing
#   curl                  - container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies are copied and installed before the source so that editing code
# does not invalidate the (very large) pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as a non-root user. The cache and data directories must be writable by
# that user, so create and chown them before dropping privileges.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser web/ ./web/
COPY --chown=appuser:appuser scripts/ ./scripts/

USER appuser

# Documentation only; compose publishes the ports it actually needs.
EXPOSE 8000

# Serves both the API and the web client.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
