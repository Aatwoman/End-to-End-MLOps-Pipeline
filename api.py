"""
api.py
FastAPI service that loads the trained model and serves predictions.

Run locally:
    uvicorn api:app --reload

Endpoints:
    GET  /health              -> liveness check
    GET  /model-info           -> metadata about the currently loaded model
    POST /predict              -> single prediction
    POST /predict-batch         -> batch predictions
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.joblib")
METADATA_PATH = Path("models/metadata.json")

_model = None
_metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _metadata
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No trained model found at {MODEL_PATH}. Run `python train.py` first."
        )
    _model = joblib.load(MODEL_PATH)
    _metadata = json.loads(METADATA_PATH.read_text())
    yield
    _model = None
    _metadata = None


app = FastAPI(
    title="Iris Classifier API",
    description="Serves predictions from a scikit-learn RandomForest trained on the Iris dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., json_schema_extra={"example": 5.1}, description="Sepal length in cm")
    sepal_width: float = Field(..., json_schema_extra={"example": 3.5}, description="Sepal width in cm")
    petal_length: float = Field(..., json_schema_extra={"example": 1.4}, description="Petal length in cm")
    petal_width: float = Field(..., json_schema_extra={"example": 0.2}, description="Petal width in cm")


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int
    class_probabilities: dict


def _features_to_dataframe(features_list: List[IrisFeatures]) -> pd.DataFrame:
    """
    Build a DataFrame with the exact column names/order used at training time.
    Passing a DataFrame (not a bare list) avoids sklearn's "X does not have
    valid feature names" warning and protects against silent column-order bugs.
    """
    columns = _metadata["feature_columns"]
    rows = [
        [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]
        for f in features_list
    ]
    return pd.DataFrame(rows, columns=columns)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model-info")
def model_info():
    if _metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return _metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    row = _features_to_dataframe([features])
    pred_index = int(_model.predict(row)[0])
    probabilities = _model.predict_proba(row)[0]
    class_names = _metadata["class_names"]

    return PredictionResponse(
        predicted_class=class_names[pred_index],
        predicted_class_index=pred_index,
        class_probabilities={name: float(p) for name, p in zip(class_names, probabilities)},
    )


@app.post("/predict-batch", response_model=List[PredictionResponse])
def predict_batch(features_list: List[IrisFeatures]):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    rows = _features_to_dataframe(features_list)
    pred_indices = _model.predict(rows)
    probabilities = _model.predict_proba(rows)
    class_names = _metadata["class_names"]

    return [
        PredictionResponse(
            predicted_class=class_names[int(idx)],
            predicted_class_index=int(idx),
            class_probabilities={name: float(p) for name, p in zip(class_names, probs)},
        )
        for idx, probs in zip(pred_indices, probabilities)
    ]
