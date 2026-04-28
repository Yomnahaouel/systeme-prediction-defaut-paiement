"""
app_v2.py — Enhanced FastAPI Deployment for Credit Default Prediction

Endpoints:
  GET  /                  → Health check
  GET  /info              → Model info & expected features
  GET  /threshold         → Optimal threshold configuration
  POST /predict           → Single client prediction
  POST /predict/batch     → Batch predictions (multiple clients)
  POST /explain           → SHAP explanation for a prediction
  GET  /features          → Feature importance ranking

Author: 7afnawi for Hefny
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import config

# ══════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Credit Default Risk Prediction API",
    description="""
    🏦 **Credit Scoring API** — Predict loan default probability
    
    This API provides:
    - Single & batch predictions
    - SHAP-based explanations
    - Optimized business threshold
    - Feature importance ranking
    
    **Model:** CatBoost with Optuna tuning (AUC ~0.786)
    
    **Author:** 7afnawi for Hefny
    """,
    version="2.0.0",
    contact={"name": "7afnawi", "email": "hefny@example.com"},
    docs_url="/docs",
    redoc_url="/redoc"
)

def _get_allowed_origins() -> List[str]:
    """Read CORS origins from env while keeping local demo usage simple."""
    origins = os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


# CORS for frontend. Override with CORS_ALLOW_ORIGINS in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════

model = None
ensemble_model = None
selected_features = None
threshold_config = None
feature_importance = None
shap_explainer = None
startup_time = None


@app.on_event("startup")
def load_artifacts():
    """Load all model artifacts at startup."""
    global model, ensemble_model, selected_features, threshold_config
    global feature_importance, startup_time
    
    startup_time = datetime.now()
    print("\n" + "═" * 50)
    print("🚀 Starting Credit Risk API v2.0")
    print("═" * 50)
    
    model_dir = config.MODEL_DIR
    
    # 1. Load primary model (CatBoost optimized)
    model_priority = [
        "catboost_optimized.joblib",
        "best_model_v2.joblib",
        "best_model.joblib",
        "CatBoost.joblib"
    ]
    
    for model_file in model_priority:
        model_path = model_dir / model_file
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"✓ Model loaded: {model_file}")
            break
    
    if model is None:
        print("✗ No model found!")
    
    # 2. Load ensemble (optional)
    ensemble_path = model_dir / "ensemble_model.joblib"
    if ensemble_path.exists():
        ensemble_model = joblib.load(ensemble_path)
        print(f"✓ Ensemble model loaded")
    
    # 3. Load features - prefer model's own feature names for correct order
    if model is not None and hasattr(model, 'feature_names_'):
        selected_features = list(model.feature_names_)
        print(f"✓ Features from model: {len(selected_features)} features")
    else:
        features_priority = [
            "model_feature_order.json",
            "selected_features_v2.json",
            "selected_features.json"
        ]
        
        for feat_file in features_priority:
            feat_path = model_dir / feat_file
            if feat_path.exists():
                with open(feat_path) as f:
                    selected_features = json.load(f)
                print(f"✓ Features loaded: {len(selected_features)} features")
                break
    
    # 4. Load threshold config
    threshold_path = model_dir / "threshold_config.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold_config = json.load(f)
        print(f"✓ Threshold config loaded: {threshold_config.get('optimal_threshold', 0.5)}")
    else:
        threshold_config = {'optimal_threshold': 0.5, 'cost_fn': 10000, 'cost_fp': 500}
    
    # 5. Load feature importance
    importance_path = model_dir / "shap_feature_importance.csv"
    if importance_path.exists():
        feature_importance = pd.read_csv(importance_path)
        print(f"✓ Feature importance loaded")
    elif (model_dir / "feature_importance.csv").exists():
        feature_importance = pd.read_csv(model_dir / "feature_importance.csv")
        print(f"✓ Feature importance loaded (fallback)")
    
    print("═" * 50 + "\n")


# ══════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════

class ClientData(BaseModel):
    """Input: Single client features."""
    features: Dict[str, Optional[float]] = Field(
        ...,
        min_length=1,
        description="Dictionary of feature names to values"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, features: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        for name, value in features.items():
            if value is not None and not np.isfinite(value):
                raise ValueError(f"Feature '{name}' must be a finite number")
        return features
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "EXT_SOURCE_2": 0.6,
                    "EXT_SOURCE_3": 0.5,
                    "EXT_SOURCE_1": 0.4,
                    "DAYS_BIRTH": -15000,
                    "AMT_CREDIT": 500000,
                    "AMT_INCOME_TOTAL": 150000,
                    "AMT_ANNUITY": 25000,
                }
            }
        }


class BatchClientData(BaseModel):
    """Input: Multiple clients."""
    clients: List[Dict[str, Optional[float]]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of client feature dictionaries"
    )

    @field_validator("clients")
    @classmethod
    def validate_clients(cls, clients: List[Dict[str, Optional[float]]]) -> List[Dict[str, Optional[float]]]:
        for index, features in enumerate(clients, start=1):
            if not features:
                raise ValueError(f"Client #{index} must contain at least one feature")
            for name, value in features.items():
                if value is not None and not np.isfinite(value):
                    raise ValueError(f"Client #{index} feature '{name}' must be a finite number")
        return clients


class PredictionResponse(BaseModel):
    """Output: Prediction result."""
    client_id: str
    default_probability: float
    prediction: int
    risk_level: str
    threshold: float
    confidence: str


class BatchPredictionResponse(BaseModel):
    """Output: Batch prediction results."""
    predictions: List[PredictionResponse]
    total_clients: int
    high_risk_count: int
    processing_time_ms: float


class ExplanationResponse(BaseModel):
    """Output: SHAP explanation."""
    base_value: float
    prediction_value: float
    top_positive_factors: List[Dict]
    top_negative_factors: List[Dict]
    risk_summary: str


class ThresholdInfo(BaseModel):
    """Output: Threshold configuration."""
    optimal_threshold: float
    cost_false_negative: float
    cost_false_positive: float
    metrics: Dict


class FeatureImportanceResponse(BaseModel):
    """Output: Feature importance list."""
    features: List[Dict]
    total_features: int


# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def prepare_features(features_dict: Dict) -> pd.DataFrame:
    """Prepare feature vector from input dict."""
    if not selected_features:
        raise HTTPException(status_code=503, detail="Feature list not loaded")

    input_dict = {}
    for feat in selected_features:
        value = features_dict.get(feat, 0.0)
        input_dict[feat] = 0.0 if value is None else value
    
    df = pd.DataFrame([input_dict])
    return df.fillna(0)


def get_risk_level(proba: float, threshold: float) -> tuple:
    """Get risk level and confidence from probability."""
    if proba < 0.15:
        return "LOW", "High"
    elif proba < 0.30:
        return "LOW-MEDIUM", "Medium"
    elif proba < threshold:
        return "MEDIUM", "Medium"
    elif proba < 0.65:
        return "HIGH", "Medium"
    elif proba < 0.80:
        return "HIGH", "High"
    else:
        return "VERY HIGH", "High"


def is_ready() -> bool:
    """The API is ready for predictions only when model and feature order are loaded."""
    return model is not None and bool(selected_features)


# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def health_check():
    """Liveness check with system status.

    This endpoint answers when the API process is alive. Use /ready for
    deployment health checks that must fail when the model is missing.
    """
    uptime = (datetime.now() - startup_time).total_seconds() if startup_time else 0
    ready = is_ready()
    
    return {
        "status": "healthy" if ready else "degraded",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "features_loaded": bool(selected_features),
        "ensemble_available": ensemble_model is not None,
        "features_count": len(selected_features) if selected_features else 0,
        "threshold": threshold_config.get('optimal_threshold', 0.5) if threshold_config else 0.5,
        "uptime_seconds": round(uptime, 2)
    }


@app.get("/ready", tags=["Health"])
def readiness_check():
    """Readiness check for Docker/Kubernetes deployments."""
    if not is_ready():
        raise HTTPException(
            status_code=503,
            detail="API not ready: trained model and feature list must be available"
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "features_loaded": True,
        "features_count": len(selected_features),
    }


@app.get("/info", tags=["Info"])
def model_info():
    """Get model information and configuration."""
    return {
        "model_type": type(model).__name__ if model else None,
        "model_auc": threshold_config.get('metrics', {}).get('auc') if threshold_config else None,
        "n_features": len(selected_features) if selected_features else 0,
        "optimal_threshold": threshold_config.get('optimal_threshold', 0.5) if threshold_config else 0.5,
        "sample_features": selected_features[:10] if selected_features else [],
        "risk_levels": ["LOW", "LOW-MEDIUM", "MEDIUM", "HIGH", "VERY HIGH"],
        "version": "2.0.0"
    }


@app.get("/threshold", response_model=ThresholdInfo, tags=["Info"])
def get_threshold():
    """Get optimal threshold configuration."""
    if not threshold_config:
        return ThresholdInfo(
            optimal_threshold=0.5,
            cost_false_negative=10000,
            cost_false_positive=500,
            metrics={}
        )
    
    return ThresholdInfo(
        optimal_threshold=threshold_config.get('optimal_threshold', 0.5),
        cost_false_negative=threshold_config.get('cost_fn', 10000),
        cost_false_positive=threshold_config.get('cost_fp', 500),
        metrics=threshold_config.get('metrics', {})
    )


@app.get("/features", response_model=FeatureImportanceResponse, tags=["Info"])
def get_feature_importance(top_n: int = 20):
    """Get top N most important features."""
    if feature_importance is None:
        raise HTTPException(status_code=503, detail="Feature importance not available")
    if top_n < 1 or top_n > 100:
        raise HTTPException(status_code=422, detail="top_n must be between 1 and 100")
    
    top_features = feature_importance.head(top_n).to_dict('records')
    
    return FeatureImportanceResponse(
        features=top_features,
        total_features=len(feature_importance)
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(data: ClientData, use_ensemble: bool = False):
    """
    Predict default probability for a single client.
    
    - **features**: Dictionary of feature name → value
    - **use_ensemble**: Use ensemble model if available (default: False)
    """
    if not is_ready():
        raise HTTPException(status_code=503, detail="Model or feature list not loaded")
    
    # Prepare features
    df = prepare_features(data.features)
    
    # Select model
    prediction_model = ensemble_model if (use_ensemble and ensemble_model) else model
    
    # Predict
    try:
        proba = prediction_model.predict_proba(df)[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    
    # Get threshold
    threshold = threshold_config.get('optimal_threshold', 0.5) if threshold_config else 0.5
    prediction = int(proba >= threshold)
    risk_level, confidence = get_risk_level(proba, threshold)
    
    return PredictionResponse(
        client_id="client_001",
        default_probability=round(float(proba), 6),
        prediction=prediction,
        risk_level=risk_level,
        threshold=threshold,
        confidence=confidence
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
def predict_batch(data: BatchClientData, use_ensemble: bool = False):
    """
    Predict default probability for multiple clients.
    
    - **clients**: List of feature dictionaries
    - **use_ensemble**: Use ensemble model if available
    """
    if not is_ready():
        raise HTTPException(status_code=503, detail="Model or feature list not loaded")
    
    import time
    start_time = time.time()
    
    predictions = []
    threshold = threshold_config.get('optimal_threshold', 0.5) if threshold_config else 0.5
    prediction_model = ensemble_model if (use_ensemble and ensemble_model) else model
    
    for i, client_features in enumerate(data.clients):
        df = prepare_features(client_features)
        
        try:
            proba = prediction_model.predict_proba(df)[:, 1][0]
            pred = int(proba >= threshold)
            risk_level, confidence = get_risk_level(proba, threshold)
            
            predictions.append(PredictionResponse(
                client_id=f"client_{i+1:03d}",
                default_probability=round(float(proba), 6),
                prediction=pred,
                risk_level=risk_level,
                threshold=threshold,
                confidence=confidence
            ))
        except Exception as e:
            predictions.append(PredictionResponse(
                client_id=f"client_{i+1:03d}_ERROR",
                default_probability=-1.0,
                prediction=-1,
                risk_level="ERROR",
                threshold=threshold,
                confidence="N/A"
            ))
    
    processing_time = (time.time() - start_time) * 1000
    high_risk = sum(1 for p in predictions if p.prediction == 1)
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_clients=len(predictions),
        high_risk_count=high_risk,
        processing_time_ms=round(processing_time, 2)
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
def explain_prediction(data: ClientData):
    """
    Get SHAP explanation for a prediction.
    
    Returns top factors contributing to the prediction.
    """
    if not is_ready():
        raise HTTPException(status_code=503, detail="Model or feature list not loaded")
    
    # Prepare features
    df = prepare_features(data.features)
    
    # Get prediction first
    proba = model.predict_proba(df)[:, 1][0]
    threshold = threshold_config.get('optimal_threshold', 0.5) if threshold_config else 0.5
    
    # Try SHAP explanation
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        
        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get contributions
        contributions = list(zip(selected_features, shap_values[0]))
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Top positive (increase risk)
        positive = [{"feature": f, "impact": round(float(v), 4)} 
                   for f, v in contributions if v > 0][:5]
        
        # Top negative (decrease risk)
        negative = [{"feature": f, "impact": round(float(v), 4)} 
                   for f, v in sorted(contributions, key=lambda x: x[1])[:5]]
        
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
        
        # Risk summary
        if proba < threshold:
            summary = f"This client has LOW default risk ({proba:.1%}). "
        else:
            summary = f"This client has HIGH default risk ({proba:.1%}). "
        
        if positive:
            summary += f"Main risk factors: {positive[0]['feature']}"
            if len(positive) > 1:
                summary += f", {positive[1]['feature']}"
        
        return ExplanationResponse(
            base_value=round(float(base_value), 4),
            prediction_value=round(float(proba), 4),
            top_positive_factors=positive,
            top_negative_factors=negative,
            risk_summary=summary
        )
        
    except ImportError:
        # SHAP not available - return simplified explanation
        risk_level, _ = get_risk_level(proba, threshold)
        return ExplanationResponse(
            base_value=0.08,  # Approximate baseline (8% default rate)
            prediction_value=round(float(proba), 4),
            top_positive_factors=[],
            top_negative_factors=[],
            risk_summary=f"Risk level: {risk_level} (probability: {proba:.1%}). SHAP not available for detailed explanation."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
