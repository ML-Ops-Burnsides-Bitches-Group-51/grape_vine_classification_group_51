# app.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict


# -----------------------------
# 1) Pydantic request/response
# -----------------------------
class PredictRequest(BaseModel):
    """
    Example schema for tabular models.
    Replace/extend feature fields to match your model inputs.
    """
    model_config = ConfigDict(extra="forbid")  # reject unknown fields

    features: List[float] = Field(..., min_length=1, description="Numeric feature vector")
    request_id: Optional[str] = Field(None, description="Optional client-provided id")


class PredictResponse(BaseModel):
    request_id: Optional[str]
    prediction: float
    model_version: str


class ModelInfo(BaseModel):
    name: str
    version: str
    trained_at: str
    metrics: Dict[str, Any]


# -----------------------------
# 2) Model wrapper + loader
# -----------------------------
@dataclass
class LoadedModel:
    name: str
    version: str
    trained_at: datetime
    metrics: Dict[str, Any]
    model: Any  # your actual model object (sklearn, torch module, etc.)

class DummyModel:
    def predict_one(self, x: List[float]) -> float:
        # Replace with real inference.
        # Example "prediction": average of features
        return sum(x) / len(x)

def load_model() -> LoadedModel:
    # Replace with: joblib.load("model.joblib") or torch.load(...) etc.
    return LoadedModel(
        name="demo-regressor",
        version="0.1.0",
        trained_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        metrics={"rmse": 1.23, "r2": 0.87},
        model=DummyModel(),
    )


# -----------------------------
# 3) FastAPI app
# -----------------------------
app = FastAPI(
    title="ML Inference API",
    version="0.1.0",
    description="A minimal FastAPI template for ML inference",
)

LOADED = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    return ModelInfo(
        name=LOADED.name,
        version=LOADED.version,
        trained_at=LOADED.trained_at.isoformat(),
        metrics=LOADED.metrics,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        pred = float(LOADED.model.predict_one(req.features))
    except Exception as e:
        # Keep errors clean for clients
        raise HTTPException(status_code=400, detail=f"Inference failed: {type(e).__name__}")

    return PredictResponse(
        request_id=req.request_id,
        prediction=pred,
        model_version=LOADED.version,
    )
