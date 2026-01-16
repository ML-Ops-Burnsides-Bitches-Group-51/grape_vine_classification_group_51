# api.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import io
import os
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
from torchvision import transforms

from grape_vine_classification.model_lightning import SimpleCNN


# -----------------------------
# 1) Pydantic response models
# -----------------------------
class PredictResponse(BaseModel):
    request_id: Optional[str] = None
    predicted_index: int
    predicted_label: str
    probabilities: List[float]
    model_version: str
    latency_ms: int


class ModelInfo(BaseModel):
    name: str
    version: str
    trained_at: str
    metrics: Dict[str, Any]
    device: str
    num_classes: int
    labels: List[str]


# -----------------------------
# 2) Model wrapper + loader
# -----------------------------
@dataclass(frozen=True)
class LoadedModel:
    name: str
    version: str
    trained_at: datetime
    metrics: Dict[str, Any]
    model: torch.nn.Module
    device: torch.device
    labels: List[str]


def _get_device() -> torch.device:
    # Same device logic style as your train.py
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_labels() -> List[str]:
    """
    Your model outputs 5 logits (see fc1 out_features=5 in model.py),
    so you must provide 5 class labels in the same order as training.
    """
    # Option A: set labels via env var:
    #   export LABELS="ClassA,ClassB,ClassC,ClassD,ClassE"
    env = os.getenv("LABELS")
    if env:
        labels = [x.strip() for x in env.split(",") if x.strip()]
        if len(labels) == 5:
            return labels

    # Option B: replace these with the real folder names / class names
    return ["class_0", "class_1", "class_2", "class_3", "class_4"]


def load_model() -> LoadedModel:
    base_dir = Path(__file__).resolve().parent.parent.parent
    weights_path = base_dir / "models" / "model.pth"
    if not weights_path.exists():
        raise RuntimeError(f"Model weights not found at: {weights_path}")

    device = _get_device()
    model = SimpleCNN().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    labels = _default_labels()
    if len(labels) != 5:
        raise RuntimeError(f"Expected 5 labels, got {len(labels)}")

    return LoadedModel(
        name="grapevine-leaf-cnn",
        version=os.getenv("MODEL_VERSION", "0.1.0"),
        trained_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        metrics={"note": "add metrics if you have them (e.g., test acc)"},
        model=model,
        device=device,
        labels=labels,
    )


# Match your preprocessing in data.py: grayscale(1) -> tensor -> resize(128,128)
# (Keeping same order you used to avoid accidental distribution shift.)
TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ]
)


# -----------------------------
# 3) FastAPI app
# -----------------------------
app = FastAPI(
    title="Grapevine Leaf Classifier API",
    version=os.getenv("API_VERSION", "0.1.0"),
    description="FastAPI for grapevine leaf image classification",
)


@app.on_event("startup")
def _startup() -> None:
    try:
        app.state.loaded = load_model()
    except Exception as e:
        # Don't crash import; but service should report unhealthy
        app.state.loaded = None
        app.state.load_error = f"{type(e).__name__}: {e}"


def _get_loaded() -> LoadedModel:
    loaded = getattr(app.state, "loaded", None)
    if loaded is None:
        err = getattr(app.state, "load_error", "unknown")
        raise HTTPException(status_code=503, detail=f"Model not loaded: {err}")
    return loaded


@app.get("/health")
def health() -> Dict[str, Any]:
    loaded = getattr(app.state, "loaded", None)
    if loaded is None:
        return {"status": "degraded", "model_loaded": False, "error": getattr(app.state, "load_error", None)}
    return {"status": "ok", "model_loaded": True}


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    m = _get_loaded()
    return ModelInfo(
        name=m.name,
        version=m.version,
        trained_at=m.trained_at.isoformat(),
        metrics=m.metrics,
        device=str(m.device),
        num_classes=len(m.labels),
        labels=m.labels,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Image file (jpg/png/etc.)"),
    request_id: Optional[str] = None,
) -> PredictResponse:
    m = _get_loaded()

    t0 = time.perf_counter()
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        x = TRANSFORM(img).unsqueeze(0).to(m.device)  # shape: [1, 1, 128, 128]
        with torch.no_grad():
            logits = m.model(x)  # shape: [1, 5]
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

        pred_idx = int(torch.tensor(probs).argmax().item())
        pred_label = m.labels[pred_idx]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {type(e).__name__}") from e

    latency_ms = int((time.perf_counter() - t0) * 1000)

    return PredictResponse(
        request_id=request_id,
        predicted_index=pred_idx,
        predicted_label=pred_label,
        probabilities=[float(p) for p in probs],
        model_version=m.version,
        latency_ms=latency_ms,
    )

# To run the app, use:
# uvicorn --reload --port 8000 src.grape_vine_classification.api:app -- host