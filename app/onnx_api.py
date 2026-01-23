from __future__ import annotations

import os
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

import onnxruntime as ort
import numpy as np


# To run the app, use:
# uvicorn --reload --port 8000 src.grape_vine_classification.onnx_api:app

# Url:
# http://localhost:8000/docs#/

# ----------------------------
# Paths that match repo
# ----------------------------
# This file lives at: src/grape_vine_classification/api.py

PKG_DIR = Path(__file__).resolve().parent 
# Fallback: if we are at /app, REPO_ROOT is just /app
try:
    REPO_ROOT = PKG_DIR.parents[0]
except IndexError:
    REPO_ROOT = PKG_DIR 
MODELS_DIR = REPO_ROOT / "models"

# You can override these with env vars when deploying
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "model.onnx")))
LABELS_PATH = Path(os.getenv("LABELS_PATH", str(MODELS_DIR / "labels.json")))

# From your README: you convert to B/W and downsize to 128x128
IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
TOP_K_DEFAULT = int(os.getenv("#_TOP_PREDS_DEFAULT", "3"))

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# ----------------------------
# Response schema
# ----------------------------
class PredictionResponse(BaseModel):
    filename: str
    predicted_label: str
    confidence: float
    probabilities: List[float]
    top_k: Optional[List[Dict[str, Any]]] = None


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


_ort_session: Optional[ort.InferenceSession] = None
_input_name: Optional[str] = None
_output_name: Optional[str] = None
_labels: List[str] = []

# Grayscale + 128x128 as described in README

def preprocess_pil(img: Image.Image) -> np.ndarray:
    """
    Convert PIL image to model input:
    - RGB → grayscale
    - resize to IMG_SIZE x IMG_SIZE
    - normalize to [0,1]
    - shape: (1, 1, H, W)
    """
    img = img.convert("L")  # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W)
    arr = arr[np.newaxis, np.newaxis, :, :]          # (1, 1, H, W)

    return arr

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
    global _ort_session, _input_name, _output_name, _labels

    _labels = _load_labels(LABELS_PATH)

    if not MODEL_PATH.exists():
        raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")

    # Choose execution providers
    providers = ["CPUExecutionProvider"]
    if ort.get_device() == "GPU":
        providers.insert(0, "CUDAExecutionProvider")

    _ort_session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=providers,
    )

    # Cache input / output names (important!)
    _input_name = _ort_session.get_inputs()[0].name
    _output_name = _ort_session.get_outputs()[0].name

def predict_pil(
    img: Image.Image, top_k: int
) -> Tuple[str, float, List[float], List[Tuple[str, float]]]:
    if _ort_session is None:
        raise RuntimeError("Model not loaded")

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Torch preprocessing → NumPy
    x = preprocess_pil(img)

    outputs = _ort_session.run(
        [_output_name],
        {_input_name: x},
    )

    logits = outputs[0][0]  # shape: (num_classes,)

    # Softmax (manual, since we’re out of torch)
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    pred_label = _labels[idx] if _labels and idx < len(_labels) else str(idx)

    top_idxs = probs.argsort()

    top_list = []
    for i in top_idxs:
        label = _labels[i] if _labels and i < len(_labels) else str(i)
        top_list.append((label, float(probs[i])))
   
    return pred_label, conf, top_list, probs

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
        "providers": _ort_session.get_providers() if _ort_session else [],
        "model_path": str(MODEL_PATH),
        "labels_path": str(LABELS_PATH),
        "num_labels": len(_labels),
        "img_size": IMG_SIZE,
    }

@app.post("/predict", response_model=BatchPredictionResponse)
async def predict(
    files: List[UploadFile] = File(...),
    num_predictions: int = 5,
):
    results: List[PredictionResponse] = []

    for file in files:
        # --- validate ---
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {file.content_type}",
            )

        try:
            data = await file.read()
            img = Image.open(BytesIO(data))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=f"Invalid image: {file.filename}")

        # --- inference ---
        label, conf, top_list, probs = predict_pil(img, top_k=num_predictions)
        
        results.append(
            PredictionResponse(
                filename=file.filename,
                predicted_label=label,
                confidence=conf,
                probabilities = [prob for prob in probs],
                top_k=[{"label": lbl, "score": s} for lbl, s in top_list],
            )
        )

    return BatchPredictionResponse(results=results)
