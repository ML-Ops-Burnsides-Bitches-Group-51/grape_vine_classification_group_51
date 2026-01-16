from __future__ import annotations

import os
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn.functional as F
from torchvision import transforms


# ----------------------------
# Paths that match your repo
# ----------------------------
# This file lives at: src/grape_vine_classification/api.py
PKG_DIR = Path(__file__).resolve().parent                   # .../src/grape_vine_classification
REPO_ROOT = PKG_DIR.parents[1]                              # repo root (two levels up)
MODELS_DIR = REPO_ROOT / "models"

# You can override these with env vars when deploying
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "model.pth")))
LABELS_PATH = Path(os.getenv("LABELS_PATH", str(MODELS_DIR / "labels.json")))

# From your README: you convert to B/W and downsize to 128x128
IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "3"))

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


# ----------------------------
# Response schema
# ----------------------------
class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float
    top_k: Optional[List[Dict[str, Any]]] = None


# ----------------------------
# Model + preprocessing
# ----------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[torch.nn.Module] = None
_labels: List[str] = []

# Grayscale + 128x128 as described in your README
_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # -> [1, H, W] float in [0,1]
    # If you normalized during training, add it here (example):
    # transforms.Normalize(mean=[0.5], std=[0.5]),
])


def _load_labels(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # supports ["a","b"] or {"0":"a","1":"b"}
    if isinstance(data, dict):
        return [data[str(i)] for i in range(len(data))]
    return list(data)


def load_model() -> None:
    global _model, _labels

    _labels = _load_labels(LABELS_PATH)

    if not MODEL_PATH.exists():
        raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")

    # 1) Build the architecture (matches model_lightning.py)
    #    IMPORTANT: adjust this import path to where the file lives in your repo.
    #    If model_lightning.py is in src/grape_vine_classification/, this is correct:
    from grape_vine_classification.model_lightning import SimpleCNN  # <-- uses your file :contentReference[oaicite:1]{index=1}

    # config is only used for optimizers; for inference it can be minimal
    config = {"optim": "Adam", "lr": 1e-3}
    model = SimpleCNN(config)

    # 2) Load checkpoint / state_dict
    state = torch.load(MODEL_PATH, map_location="cpu")

    # If it's a Lightning checkpoint, weights are often under "state_dict"
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Some training scripts save under different keys
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {type(state)}")

    # 3) Clean common prefixes: "model.", "module."
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    model.to(_device)
    model.eval()
    _model = model


@torch.inference_mode()
def predict_pil(img: Image.Image, top_k: int) -> Tuple[str, float, List[Tuple[str, float]]]:
    if _model is None:
        raise RuntimeError("Model not loaded")

    # Convert to RGB first to avoid odd modes, then preprocess forces grayscale anyway.
    if img.mode != "RGB":
        img = img.convert("RGB")

    x = _preprocess(img).unsqueeze(0).to(_device)  # [1, 1, 128, 128] by default

    logits = _model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)

    conf, idx = torch.max(probs, dim=0)
    idx_i = int(idx.item())
    conf_f = float(conf.item())

    pred_label = _labels[idx_i] if _labels and idx_i < len(_labels) else str(idx_i)

    k = max(1, min(int(top_k), probs.numel()))
    top_probs, top_idxs = torch.topk(probs, k=k)
    top_list: List[Tuple[str, float]] = []
    for p, i in zip(top_probs.tolist(), top_idxs.tolist()):
        ii = int(i)
        label = _labels[ii] if _labels and ii < len(_labels) else str(ii)
        top_list.append((label, float(p)))

    return pred_label, conf_f, top_list


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Grape Vine Classification API", version="1.0.0")


@app.on_event("startup")
def _startup():
    try:
        load_model()
    except Exception as e:
        # fail fast so Docker / deployment catches it immediately
        raise RuntimeError(f"Startup failed loading model: {e}") from e


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(_device),
        "model_path": str(MODEL_PATH),
        "labels_path": str(LABELS_PATH),
        "num_labels": len(_labels),
        "img_size": IMG_SIZE,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), top_k: int = TOP_K_DEFAULT):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    try:
        data = await file.read()
        img = Image.open(BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file (cannot decode).")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed reading image: {e}")

    try:
        label, conf, top_list = predict_pil(img, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictionResponse(
        predicted_label=label,
        confidence=conf,
        top_k=[{"label": l, "score": s} for l, s in top_list],
    )


# To run the app, use:
# uvicorn --reload --port 8000 src.grape_vine_classification.api:app -- host