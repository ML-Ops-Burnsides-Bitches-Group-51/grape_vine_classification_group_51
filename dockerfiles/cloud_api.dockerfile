FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies (consistent with training)
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

# UV configurations
ENV UV_LINK_MODE=copy


# Install enviroment
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy functions
COPY src/ src/

EXPOSE 8011

CMD exec uv run uvicorn grape_vine_classification.cloud_api:app --app-dir src --port ${PORT:-8011} --host 0.0.0.0

# CMD ["uv", "run", "uvicorn", "cloud_api:app", "--host", "0.0.0.0", "--port", "8080"]

# docker run -it -p 8080:8080 \
#   -e PORT=8080 \
#   -e MODEL_BUCKET=models_grape_gang \
#   -e GOOGLE_CLOUD_PROJECT=grapevine-gang \
#   -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/creds.json \
#   -v ~/.config/gcloud/application_default_credentials.json:/tmp/keys/creds.json:ro \
#   cloud_api:latest
