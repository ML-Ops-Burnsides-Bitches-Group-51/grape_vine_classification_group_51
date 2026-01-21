# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /

# Set environment variables
ENV UV_LINK_MODE=copy

# Copy dependencies
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

# Copy source code
COPY src/ src/
# COPY data/ data/ # We don't want to copy the data
# COPY models/ models/
COPY configs/ configs/

#RUN uv sync --frozen --no-install-project

RUN --mount=type=cache,target=/root/.cache/uv uv sync


ENTRYPOINT ["uv", "run", "src/grape_vine_classification/train_lightning.py"]

CMD [ \
    "--config-path", "configs/experiment/exp1.yaml", \
    "--data-path", "/data/processed_dataset", \
    "--model-path", "/models/docker_model.pth" \
]