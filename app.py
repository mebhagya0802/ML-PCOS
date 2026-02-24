# app.py
from fastapi import FastAPI, HTTPException # For raising HTTP errors (e.g., 400, 503).
from pydantic import BaseModel, conlist, Field # For data validation and serialization.
from typing import List, Literal, Optional
import joblib
import numpy as np # For array operations during inference.
from starlette.responses import JSONResponse # For custom HTTP responses.

#FastAPI App Initialization
app = FastAPI(
    title="Iris ML REST API",
    version="1.0.0",
    description="FastAPI service wrapping a scikit-learn model (RandomForest) for Iris classification."
)

# ---- Load model at startup ----
MODEL_PATH = "model/iris_rf.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"WARNING: Could not load model at startup: {e}")

# ---- Schemas ----

class IrisFeatures(BaseModel):
    features: List[float] = Field(
        ..., 
        min_length=4, 
        max_length=4, 
        description="Four numeric features for Iris prediction"
    )
    request_id: Optional[str] = Field(default=None, example="req-12345")


class BatchIrisFeatures(BaseModel):
    inputs: List[List[float]] = Field(
        ..., 
        min_length=1,
        description="Batch of Iris features, each must be exactly 4 numbers",
        example=[[5.1, 3.5, 1.4, 0.2], [6.0, 2.2, 5.0, 1.5]]
    )
    request_id: Optional[str] = Field(default=None, example="batch-777")

    # Custom validator to enforce exactly 4 features
    @classmethod
    def validate_inputs(cls, value):
        for row in value:
            if len(row) != 4:
                raise ValueError("Each input must have exactly 4 features")
        return value

class Prediction(BaseModel):
    label: Literal["setosa", "versicolor", "virginica"]
    proba: float
    request_id: Optional[str] = None

class BatchPrediction(BaseModel):
    predictions: List[Prediction]
    request_id: Optional[str] = None

id_to_label = {0: "setosa", 1: "versicolor", 2: "virginica"}

# ---- Endpoints ----
@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/ready", tags=["ops"])
def ready():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready"}

@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(payload: IrisFeatures):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    x = np.array(payload.features, dtype=float).reshape(1, -1)
    try:
        pred_id = int(model.predict(x)[0])
        proba = float(np.max(model.predict_proba(x)))
        return Prediction(label=id_to_label[pred_id], proba=round(proba, 6), request_id=payload.request_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

@app.post("/predict-batch", response_model=BatchPrediction, tags=["inference"])
def predict_batch(payload: BatchIrisFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = np.array(payload.inputs, dtype=float)
    try:
        pred_ids = model.predict(X)
        probas = model.predict_proba(X).max(axis=1)
        preds = [
            Prediction(label=id_to_label[int(i)], proba=round(float(p), 6), request_id=payload.request_id)
            for i, p in zip(pred_ids, probas)
        ]
        return BatchPrediction(predictions=preds, request_id=payload.request_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch inference error: {e}")
