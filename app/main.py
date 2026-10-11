"""
Minimal FastAPI service for serving a scikit-learn model.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="DS Model API", version="0.1.0")

MODEL_PATH = Path(__file__).parent.parent / "models" / "example_model.joblib"

# Create a dummy model if none exists (for first run)
if not MODEL_PATH.exists():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    model = LogisticRegression().fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    features: list[float]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None = None


@app.get("/")
def health():
    return {"status": "ok", "message": "Model API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    X = np.array(req.features).reshape(1, -1)
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    return PredictionResponse(prediction=pred, probability=proba)
