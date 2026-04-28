"""
Model artifact tests.

These tests validate a real trained `.joblib` model when it is available.
If no model artifact is present, they are skipped instead of failing the
whole CI run, because large trained binaries are often excluded from Git.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PRIORITY = [
    "catboost_optimized.joblib",
    "best_model_v2.joblib",
    "best_model.joblib",
    "CatBoost.joblib",
]


def find_model_path():
    for name in MODEL_PRIORITY:
        path = MODEL_DIR / name
        if path.exists():
            return path
    return None


@pytest.fixture(scope="module")
def model_path():
    path = find_model_path()
    if path is None:
        pytest.skip("No trained .joblib model artifact found in models/")
    return path


@pytest.fixture(scope="module")
def model(model_path):
    return joblib.load(model_path)


@pytest.fixture(scope="module")
def features(model):
    if hasattr(model, "feature_names_"):
        return list(model.feature_names_)

    for filename in ["model_feature_order.json", "selected_features_v2.json", "selected_features.json"]:
        feat_path = MODEL_DIR / filename
        if feat_path.exists():
            with open(feat_path) as f:
                return json.load(f)

    pytest.skip("No feature-order metadata found")


class TestModelLoading:
    def test_model_file_exists(self, model_path):
        assert model_path.exists()

    def test_model_loads_successfully(self, model):
        assert model is not None

    def test_model_has_predict_proba(self, model):
        assert hasattr(model, "predict_proba")


class TestModelPredictions:
    def test_prediction_returns_probability(self, model, features):
        X = pd.DataFrame({feat: [0.5] for feat in features})
        proba = model.predict_proba(X)
        assert proba.shape[1] == 2
        assert 0 <= proba[0][1] <= 1

    def test_prediction_range(self, model, features):
        n_samples = 10
        X = pd.DataFrame({feat: np.random.random(n_samples) for feat in features})
        proba = model.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_prediction_sums_to_one(self, model, features):
        X = pd.DataFrame({feat: [0.5] for feat in features})
        proba = model.predict_proba(X)
        assert np.isclose(proba[0].sum(), 1.0)


class TestFeatureImportance:
    def test_feature_importance_exists(self):
        importance_path = MODEL_DIR / "feature_importance.csv"
        assert importance_path.exists() or len(list(MODEL_DIR.glob("*importance*"))) > 0

    def test_feature_importance_format(self):
        importance_path = MODEL_DIR / "feature_importance.csv"
        if importance_path.exists():
            df = pd.read_csv(importance_path)
            assert "feature" in df.columns
            assert "importance" in df.columns
            assert len(df) > 0
