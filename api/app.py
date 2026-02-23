"""
app.py — FastAPI Deployment for Credit Default Prediction

Endpoints:
  GET  /          → Health check
  GET  /info      → Model info & expected features
  POST /predict   → Predict default probability for a single client

Usage:
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Author: Yomna Haouel
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

import config

# ──────────────────────────────────────────────
# APP INITIALIZATION
# ──────────────────────────────────────────────

app = FastAPI(
    title="Credit Default Risk Prediction API",
    description="Prédiction de défaut de paiement — Projet Data Science",
    version="1.0.0",
    contact={"name": "Yomna Haouel"},
)

# Load model and features at startup
model = None
selected_features = None
preprocessing_pipeline = None


@app.on_event("startup")
def load_artifacts():
    """Load model, features, and pipeline at startup."""
    global model, selected_features, preprocessing_pipeline

    # Load best model
    if config.BEST_MODEL_PATH.exists():
        model = joblib.load(config.BEST_MODEL_PATH)
        print(f"✓ Model loaded: {config.BEST_MODEL_PATH.name}")
    else:
        print(f"✗ Model not found: {config.BEST_MODEL_PATH}")

    # Load selected features
    if config.SELECTED_FEATURES_PATH.exists():
        with open(config.SELECTED_FEATURES_PATH) as f:
            selected_features = json.load(f)
        print(f"✓ Features loaded: {len(selected_features)} features")
    else:
        print(f"✗ Features file not found")

    # Load preprocessing pipeline
    pipeline_path = config.MODEL_DIR / "preprocessing_pipeline.joblib"
    if pipeline_path.exists():
        preprocessing_pipeline = joblib.load(pipeline_path)
        print(f"✓ Preprocessing pipeline loaded")


# ──────────────────────────────────────────────
# SCHEMAS
# ──────────────────────────────────────────────

class ClientData(BaseModel):
    """Input schema: client financial data as key-value pairs."""
    features: Dict[str, Optional[float]]

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "AMT_CREDIT": 500000.0,
                    "AMT_INCOME_TOTAL": 150000.0,
                    "AMT_ANNUITY": 25000.0,
                    "AMT_GOODS_PRICE": 450000.0,
                    "DAYS_BIRTH": -15000,
                    "DAYS_EMPLOYED": -2000,
                    "EXT_SOURCE_1": 0.5,
                    "EXT_SOURCE_2": 0.6,
                    "EXT_SOURCE_3": 0.4,
                    "CNT_FAM_MEMBERS": 3.0,
                }
            }
        }


class PredictionResponse(BaseModel):
    """Output schema."""
    client_id: str
    default_probability: float
    prediction: int
    risk_level: str
    threshold: float


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": selected_features is not None,
    }


@app.get("/info")
def model_info():
    """Return model info and expected features."""
    return {
        "model_type": type(model).__name__ if model else None,
        "n_features": len(selected_features) if selected_features else 0,
        "expected_features": selected_features[:20] if selected_features else [],
        "note": "Send all features in /predict. Missing features will be filled with 0.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: ClientData):
    """
    Predict default probability for a single client.

    Receives client features as a dict, returns default probability
    and risk classification.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if selected_features is None:
        raise HTTPException(status_code=503, detail="Feature list not loaded")

    # Build feature vector
    input_dict = {}
    for feat in selected_features:
        input_dict[feat] = data.features.get(feat, 0.0)

    df = pd.DataFrame([input_dict])

    # Fill NaN with 0 (consistent with training)
    df = df.fillna(0)

    # Predict
    try:
        proba = model.predict_proba(df)[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    # Threshold-based classification
    threshold = 0.5
    prediction = int(proba >= threshold)

    # Risk level
    if proba < 0.2:
        risk_level = "LOW"
    elif proba < 0.5:
        risk_level = "MEDIUM"
    elif proba < 0.8:
        risk_level = "HIGH"
    else:
        risk_level = "VERY HIGH"

    return PredictionResponse(
        client_id="client_request",
        default_probability=round(float(proba), 6),
        prediction=prediction,
        risk_level=risk_level,
        threshold=threshold,
    )


# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
