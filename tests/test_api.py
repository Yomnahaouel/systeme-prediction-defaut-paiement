"""
API Tests - Unit tests for the Credit Risk API.

These tests use FastAPI's TestClient instead of requiring a running
localhost server, so they can run reliably in CI and local development.
"""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import api.app_v2 as app_module

client = TestClient(app_module.app)


class DummyModel:
    """Minimal model stub with a predict_proba API."""

    def predict_proba(self, X):
        # Higher risk when credit/income and annuity/income are high,
        # lower risk when external scores are high.
        credit = float(X.get("AMT_CREDIT", pd.Series([0])).iloc[0])
        income = max(float(X.get("AMT_INCOME_TOTAL", pd.Series([1])).iloc[0]), 1.0)
        annuity = float(X.get("AMT_ANNUITY", pd.Series([0])).iloc[0])
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in X]
        ext_mean = float(X[ext_cols].mean(axis=1).iloc[0]) if ext_cols else 0.5
        score = 0.12 + 0.04 * (credit / income) + 0.6 * (annuity / income) - 0.25 * ext_mean
        proba = min(max(score, 0.01), 0.99)
        return np.array([[1 - proba, proba]])


def setup_module():
    app_module.model = DummyModel()
    app_module.ensemble_model = None
    app_module.selected_features = [
        "AMT_CREDIT",
        "AMT_INCOME_TOTAL",
        "AMT_ANNUITY",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
    ]
    app_module.threshold_config = {
        "optimal_threshold": 0.35,
        "cost_fn": 10000,
        "cost_fp": 500,
        "metrics": {"auc": 0.786},
    }
    app_module.feature_importance = pd.DataFrame(
        {
            "feature": ["EXT_SOURCE_2", "AMT_CREDIT"],
            "importance": [0.5, 0.3],
        }
    )


class TestAPIHealth:
    def test_health_check(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["features_loaded"] is True

    def test_readiness_check(self):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestPredictions:
    def test_predict_valid_input(self):
        features = {
            "AMT_CREDIT": 500000,
            "AMT_INCOME_TOTAL": 150000,
            "AMT_ANNUITY": 25000,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.4,
        }
        response = client.post("/predict", json={"features": features})
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["default_probability"] <= 1
        assert data["prediction"] in [0, 1]
        assert data["risk_level"] in ["LOW", "LOW-MEDIUM", "MEDIUM", "HIGH", "VERY HIGH"]

    def test_predict_rejects_empty_features(self):
        response = client.post("/predict", json={"features": {}})
        assert response.status_code == 422

    def test_predict_rejects_non_numeric_features(self):
        response = client.post("/predict", json={"features": {"AMT_CREDIT": "bad"}})
        assert response.status_code == 422

    def test_predict_batch(self):
        payload = {
            "clients": [
                {"AMT_CREDIT": 200000, "AMT_INCOME_TOTAL": 300000, "AMT_ANNUITY": 10000},
                {"AMT_CREDIT": 800000, "AMT_INCOME_TOTAL": 50000, "AMT_ANNUITY": 50000},
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_clients"] == 2
        assert len(data["predictions"]) == 2

    def test_predict_batch_rejects_too_many_clients(self):
        payload = {"clients": [{"AMT_CREDIT": 100000}] * 101}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422


class TestModelInfo:
    def test_get_model_info(self):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert data["n_features"] > 0

    def test_get_feature_importance(self):
        response = client.get("/features?top_n=1")
        assert response.status_code == 200
        data = response.json()
        assert data["total_features"] == 2
        assert len(data["features"]) == 1
