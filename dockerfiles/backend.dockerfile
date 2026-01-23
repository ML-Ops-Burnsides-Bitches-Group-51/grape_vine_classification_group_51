FROM ghcr.io/astral-sh/uv:python3.12-bookworm

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_LINK_MODE=copy

COPY app/requirements_backend.txt /app/requirements_backend.txt
COPY app/onnx_api.py /app/onnx_api.py
COPY models/ /models/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache -r requirements_backend.txt

ENV MODEL_PATH=/models/model.onnx
ENV LABELS_PATH=/models/labels.json

EXPOSE 8000
CMD ["uvicorn", "onnx_api:app", "--host", "0.0.0.0", "--port", "8000"]