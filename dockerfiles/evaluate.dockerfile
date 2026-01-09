FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

#RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY data/ data/
COPY models/ models/



WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync


ENTRYPOINT ["uv", "run", "src/s2_cnn_mnist/evaluate.py"]