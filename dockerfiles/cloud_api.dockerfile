FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies (consistent with training)
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# UV configurations
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
# Add venv to path so 'uvicorn' is found automatically
ENV PATH="/app/.venv/bin:$PATH"

# Install enviroment
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy functions
COPY src/grape_vine_classification/cloud_api.py cloud_api.py
COPY src/grape_vine_classification/model_lightning.py model_lightning.py


CMD ["sh", "-c", "uvicorn cloud_api:app --host 0.0.0.0 --port ${PORT:-8080}"]